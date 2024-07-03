import common_resources as cr
import constant as co
import RnnNetworkModel
import Tokenizer
result_map={
    1:'critical', 2:'high', 3:'medium', 4:'low', 5:'open', 6:'pause', 7:'resolved', 8:'close', 9:'delete', 10:'escalate', 11:'internal', 12:'billable', 13:'respond', 14:'overdue',15:'unassign',16:'addon',17:'without addon',
}
torch,device=cr.get_torch_data()
def load_model():
    model = RnnNetworkModel.NETWORK(input_dim=co.INPUT_DIM,
            embedding_dim=co.EMBEDDING_DIM,
            hidden_dim=co.HIDDEN_DIM,
            output_dim=co.NUM_CLASSES
)

    model = model.to(device=device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9) 
    clip_value = 1.0
    model.load_state_dict(torch.load('C:\\Toothless\\model_ini_up_v_1_3.pth'))
    return model

def get_token():
    token=Tokenizer.Tokenizer(["sample"])
    token.load_tokenizer('C:\\Toothless\\tokenizer\\vocab.json','C:\\Toothless\\tokenizer\\merges.txt')
    return token

def test_data(sentence):
    model=load_model()
    token=get_token()
    sam=token.encode([sentence])[0].ids
    batch1=cr.batch_spliter(input_data_sources=[sam],out_data_sources=None)
    model.eval()
    with torch.no_grad():
        out_put=model(batch1[0].get('input'),batch1[0].get('sequence'))
        max_element=torch.max(out_put).item()
        # out_list=[1 if i > max_element-30 or i == max_element-30  else 0 for i in out_put.tolist()[0]]
        out_list=[1 if i > 0  else 0 for i in out_put.tolist()[0]]

        final_out=[]
        for i ,j in enumerate(out_list):
            if j==1:
                final_out.append(result_map.get(i+1))
        print(final_out)
        print(out_put)
test_data('show all without addon tickets')