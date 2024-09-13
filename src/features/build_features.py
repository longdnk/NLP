import pandas as pd
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

df = pd.read_csv('/kaggle/input/Dataset_articles_NoID-2.csv', encoding = "utf-8")

df = df[["Summary", "Contents"]]
df.columns = ["summary", "text"]

df = df.dropna()
df.head()

df.shape

sns.set(style='whitegrid', palette='muted', font_scale = 1.2)
rcParams['figure.figsize'] = 16, 10

pl.seed_everything(42)

train_df, test_df = train_test_split(df, test_size = 0.0001)
train_df.shape, test_df.shape

class DataSet(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: PegasusTokenizer,
            text_max_token_len: int = 512,
            summary_max_token_len: int = 300,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row["text"]
        text_encoding = tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        summary_encoding = tokenizer(
            data_row["summary"],
            max_length=self.summary_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            text=text,
            summary=data_row["summary"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten()
        )

class SummaryDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            tokenizer: PegasusTokenizer,
            batch_size: int = 8,
            text_max_token_len: int = 512,
            summary_max_token_len: int = 300
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df

        self.batch_size = batch_size
        self.tokenizer = tokenizer

        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):
        self.train_dataset = DataSet(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.test_dataset = DataSet(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

MODEL_NAME = 'google/pegasus-cnn_dailymail'
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME, token=token)

data_module = SummaryDataModule(train_df, test_df, tokenizer,batch_size = BATCH_SIZE)