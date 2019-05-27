# ChineseSegmentationPytorch
这是中文分词项目，使用pytorch框架的cnn,lstm等模型

#### 1.preprocess
data目录里是数据集

删去无用字符，打乱后 train 70% / dev 20% / test 10% 划分

#### 2.represent

序列向量化，得到 sent、label，pad() 填充为相同长度

#### 3.build

通过lstm,cnn构建序列标注模型，计算 mask_loss、mask_acc

#### 4.segment

predict() 比较原句和填充长度得到 mask_pred，在为 1 的字后插入空格
