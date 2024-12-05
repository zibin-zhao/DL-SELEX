from transformers import BertModel, BertTokenizer, BertConfig

# Load the DNABERT configuration
config = BertConfig.from_pretrained("./dnabert/6mer/config.json")

# Load the DNABERT model
model = BertModel.from_pretrained("./dnabert/6mer/pytorch_model.bin", config=config)

# Make sure to put the model in evaluation mode
model.eval()
