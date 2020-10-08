def handler_edges(edges_embed):
    embed = []
    for edge_embed in edges_embed:
        tmp = torch.sum(edge_embed,dim=2) / edge_embed.shape[1]
        embed.append(tmp.unsqueeze(0))
    return torch.cat(embed,dim=0)

def handler_graph(nodes_embed,edges_embed,edge_types):
    embed = []
    for i in range(len(nodes_embed)):
        # find speicfic edges for this graph
        edge_type = edge_types[i]
        edge_embed_cur = edges_embed[i]
        edge_type_embed = []
        for idx in edge_type:
          edge_type_embed.append(edge_embed_cur[idx])
        edge_embed = torch.cat(edge_type_embed)
        assert(edge_embed.shape[0]==edge_type.shape[0])
        tmp = torch.cat((nodes_embed[i],edge_embed),dim=0)
        tmp = torch.sum(tmp,dim=0) / tmp.shape[0]
        embed.append(tmp.unsqueeze(0))
    return torch.cat(embed,dim=0)

def top_k_logits(logits, k):
  """Mask logits so that only top-k logits remain
  """
  values, _ = torch.topk(logits, k)
  min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
  return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


top_k = 1
temperature = 0.7

# try with no regularization
class RGCNLayer(nn.Module):
    def __init__(self, num_rels, in_feat=256, out_feat=256, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False, is_output_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        # self.weight = weight # [num_rel+1,768] the last one is for self-loop
        # self.self_weight = nn.Parameter(torch.Tensor(num_nodes, out_feat)) # for self loop weight
        # print('weight.shape:',self.weight.shape)
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer
        # self.start_trans = nn.Linear(768,256)
        # self.end_trans = nn.Linear(256,768)

    def forward_comet(self, heads, tails):
        # implemented to fine-tune in comet format s + r to decode o
        # heads [bsz,768] tails [bsz]
        head_embed = self.start_trans(heads)
        rel_embed = self.weight[tails] 
        result_embed = torch.bmm(head_embed.unsqueeze(1),rel_embed)
        return self.end_trans(result_embed.squeeze(1))       

    def forward(self, g):
        # if self.num_bases < self.num_rels:
        #     # generate all weights from bases (equation (3))
        #     weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
        #     weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.in_feat, self.out_feat)

        if self.is_input_layer:
            def message_func(edges):             
                node_hidden = self.start_trans(edges.src['h']).unsqueeze(1)
                # print('node_hidden.shape:',node_hidden.shape)
                # print('edges:',edges.data['rel_type'].shape)
                # print('weight.shape:',self.weight[edges.data['rel_type'].squeeze(1)].shape)
                core_weight = torch.bmm(node_hidden,self.weight[edges.data['rel_type'].squeeze(1)])
                # print('core_weight.shape:',core_weight.shape)
                # print('norm.shape:',edges.data['norm'].unsqueeze(-1).shape)
                msg = core_weight * edges.data['norm'].unsqueeze(-1)
                del core_weight
                msg = msg.squeeze(1)
                # print('msg.shape:',msg.shape)
                return {'msg': msg}
        elif self.is_output_layer:
            def message_func(edges):
                # print('here in output')
                node_hidden = edges.src['h'].unsqueeze(1)
                core_weight = torch.bmm(node_hidden,self.weight[edges.data['rel_type'].squeeze(1)])
                msg = core_weight * edges.data['norm'].unsqueeze(-1)
                del core_weight
                msg = msg.squeeze(1)
                msg = self.end_trans(msg)
                # print('msg.shape:',msg.shape)
                return {'msg': msg}

        else:
            def message_func(edges):
                # print('here in hidden')
                node_hidden = edges.src['h'].unsqueeze(1)
                # print('node_hidden.shape:',node_hidden.shape)
                core_weight = torch.bmm(node_hidden,self.weight[edges.data['rel_type'].squeeze(1)])
                msg = core_weight * edges.data['norm'].unsqueeze(-1)
                del core_weight
                msg = msg.squeeze(1)
                # print('msg.shape:',msg.shape)
                return {'msg': msg}

        def apply_func(nodes):
            # print('here in apply')
            h = nodes.data['h']
            # print('h:',h)
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h,inplace=True)
            return {'h': h}

        def revc_func(nodes):
            # print('here in revc func')
            # print(torch.sum(nodes.mailbox['msg'], dim=1).shape)
            return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

        g.update_all(message_func, revc_func, apply_func)

class RGCNModel(nn.Module):
    def __init__(self, num_rels, num_bases=-1, num_hidden_layers=0):
        super(RGCNModel, self).__init__()
        # self.h_dim = h_dim
        # self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()
        

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def build_input_layer(self):
        return RGCNLayer(self.num_rels, activation=F.relu, is_input_layer=True)
        # return RGCNLayer(self.num_rels, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.num_rels, activation=F.relu)
        # return RGCNLayer(self.num_rels)

    def build_output_layer(self):
        return RGCNLayer(self.num_rels, activation=F.relu, is_output_layer=True)
        # return RGCNLayer(self.num_rels, is_output_layer=True)

    def forward_comet(self,heads,tails):
        # for now just use rel weight of the last layer
        results = []
        for layer in self.layers:
          results.append(layer.forward_comet(heads,tails).unsqueeze(0))
        result_embed = torch.cat(results)
        result_embed = torch.sum(result_embed,dim=0) / result_embed.shape[0]
        # print('result_embed:',result_embed.shape)
        return result_embed

    def forward(self,gs,features,edge_types,edge_norms):
        node_results = []
        edge_results = []
        for i in range(len(gs)):
          g = gs[i]
          g.ndata['h'] = features[i] # [node_num, hidden]
          g.edata['rel_type'] = edge_types[i].to(device) # [edge_num, 1]
          g.edata['norm'] = edge_norms[i].to(device) # [edge_num, 1]

          for layer in self.layers:
              layer(g)
          node_results.append(g.ndata.pop('h'))
          edge_results.append(self.layers[-1].weight)
          g.edata.pop('rel_type')
          g.edata.pop('norm')

        return node_results, edge_results


class R_GCN_GPT2(nn.Module):
    def __init__(self, num_rels, num_bases=-1, num_hidden_layers=0):
        super(R_GCN_GPT2, self).__init__()
        self.rgcn_model = RGCNModel(num_rels)
        # self.path_embedding = nn.Embedding(50257, 768)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        # self.weight_trans = nn.Linear(256,768)
        # self.node_embedding = BertModel.from_pretrained('bert-base-uncased')
        # self.path_embedding = GPT2Model.from_pretrained('gpt2')

    def get_node_embedding(self,entity_ids):
        features = []
        for entity_id_list in entity_ids:
          entity_embeds = []
          for entity_id in entity_id_list:
            entity = id2entity[entity_id]
            input_ids = torch.LongTensor(tokenizer_bert.encode(entity)).unsqueeze(0).to(device)
            entity_embed = self.node_embedding(input_ids=input_ids)[0].squeeze(0)
            entity_embed = torch.sum(entity_embed,dim=0) / entity_embed.shape[0]
            entity_embeds.append(entity_embed.unsqueeze(0))
            feature = torch.cat(entity_embeds)
          features.append(feature)
        return features

    def get_head_embedding(self,ids):
        entity_embeds = []
        for id in ids:
          entity = id2entity[id]
          input_ids = torch.LongTensor(tokenizer_bert.encode(entity)).unsqueeze(0).to(device)
          entity_embed = self.node_embedding(input_ids=input_ids)[0].squeeze(0)
          entity_embed = torch.sum(entity_embed,dim=0) / entity_embed.shape[0]
          entity_embeds.append(entity_embed.unsqueeze(0))
        return torch.cat(entity_embeds)

    def get_tail_embedding(self,ids):
        entity_embeds = []
        for id in ids:
          input_ids = torch.LongTensor(id).unsqueeze(0).to(device)
          # print('tail input_ids:',input_ids.shape)
          # entity_embed = self.gpt2_model(input_ids)[2][-1].squeeze(0)
          entity_embed = self.gpt2_model.transformer.wte(input_ids).squeeze(0)
          # print('tail entity_embed:',entity_embed.unsqueeze(0).shape)
          # entity_embed = torch.sum(entity_embed,dim=0) / entity_embed.shape[0]
          entity_embeds.append(entity_embed.unsqueeze(0))
        return torch.cat(entity_embeds)

    def get_sr_embedding(self,heads,rels):
        sr_embeds = []
        for i in range(len(heads)):
            tokens = tokenizer_gpt2.encode(head + ' ' + rel)
            # print('tokens:',tokens)
            input_ids = torch.LongTensor(tokens + (length - len(tokens)) * [50256]).unsqueeze(0).to(device)
            # sr_embedding = self.node_embedding(input_ids=input_ids)[0]
            sr_embedding = self.gpt2_model.transformer.wte(input_ids)
            mask = torch.LongTensor([1] * len(tokens) + [0] * (length - len(tokens))).unsqueeze(0).to(device)
            sr_embeds.append(sr_embedding)
            masks.append(mask)

        return torch.cat(sr_embeds)


    def generate_bert(self, batch):

        with torch.no_grad():

            # sen_ids = batch['sen_ids']
            prev_pred = batch['sr_ids']
            mask = batch['sr_mask']
            sentence = []
            past = None
            length = 1
            # decoding loop
            for i in range(20):       
                # print('mask.shape:',mask.shape)
                # print('input_embed.shape:',input_embed.shape)
                logits, past = self.gpt2_model(input_ids=prev_pred,attention_mask=mask,past=past)
                mask = F.pad(mask, (0, 1), "constant", 1.0) # add 1 to the last of the sentence mask
                # print('mask:',mask.shape)
                # logits = model.gpt2_model(attention_mask=mask,inputs_embeds=input_embed)
                # logits = logits[0]
                
                # logits, past = decoder(input_ids=prev_pred, past=past, attention_mask=mask)
                logits = logits[:,-1,:]
                # print('logits:',logits.shape)
                logits = logits.squeeze(1) / temperature
                logits = top_k_logits(logits, k=top_k)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence.append(prev_pred)
                # if prev_pred[0][0] == 50256:
                #     break
                length += 1
                # print('prev_pred:',prev_pred)
                # input_embed = self.gpt2_model.transformer.wte(prev_pred).squeeze(0)

            sentence = torch.cat(sentence, dim=-1)
            # print('sentence:',sentence.shape)

        return sentence

    def forward_bert(self,batch):
        sen_ids = batch['sen_ids']
        # sr_ids = batch['sr_ids']
        attention_mask = batch['sen_mask']
        # mask = torch.LongTensor(batch['mask']).to(device)

        
        logits = self.gpt2_model(input_ids=sen_ids,attention_mask=attention_mask)
        
        return logits

    def forward(self, batch):
        g = batch['g']
        path = batch['path']
        path_mask = batch['path_mask']
        edge_types = batch['edge_types']
        edge_norms = batch['edge_norms']
        # features = batch['features']
        names = batch['names']
        features = self.get_node_embedding(names)

        nodes_embed, rel_weights = self.rgcn_model(g,features,edge_types,edge_norms) #[bsz,node_num,hidden], [rel_num,256]
        
        

        # nodes_embed = handler_nodes(nodes_embed)    # [bsz,768]
        edges_embed = handler_edges(rel_weights)
        # edges_embed = self.edge_embed_tran(edge_feature) #[bsz,hidden]
        # edges_embed = edge_feature
        # print('nodes_embed.shape:',nodes_embed.shape)
        # print('edges_embed.shape:',edges_embed.shape)

        edges_embed = self.weight_trans(edges_embed)   # [bsz,45,768]
        # print('edges_embed.shape:',edges_embed.shape)
        
        graph_embed = handler_graph(nodes_embed,edges_embed,edge_types).unsqueeze(1)
        del nodes_embed, edges_embed
        # print('graph_embed:',graph_embed.shape)
        
        # with torch.no_grad():
        path_embed = self.gpt2_model(path)[2][-1]

        # print('path_embed:',path_embed.shape)


        # print('graph_embed:',graph_embed.shape)
        # print('path_mask:',path_mask.shape)
        input_embed = torch.cat((graph_embed,path_embed),dim=1)
        graph_embed = graph_embed.cpu()
        torch.cuda.empty_cache()

        # print('input_embed.shape:',input_embed.shape)
        ones = torch.ones(path_mask.shape[0],1).to(device)
        # print('ones.shape:',ones.shape)
        mask = torch.cat([ones,path_mask],dim=1)
        del ones
        # print('mask:',mask.shape)
        logits = self.gpt2_model(attention_mask=mask,inputs_embeds=input_embed)

        return logits