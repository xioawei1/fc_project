# 使用XGBoost进行特征提取



## 程序所使用的依赖环境

本程序的运行环境：hadoop-3.2.4 , spark-3.2.4 ,python-3.8 ,  基于yarn模式 ，运行机器数量三台。



## 程序主要思路



### 数据预处理阶段

#### pyspark部署阶段

包括：In[1] -- > In[2]

主要内容：创建一个yarn模式下的pyspark程序

### 数据处理阶段

##### 导入数据

通过代码 

```python
data=spark.read.csv(
    "/pyspark_xgboot/data.csv",inferSchema=True,header=True,encoding="gbk"
)
```

导入数据，命名为data，并将第一列保留设置为列名，data格式为dataFrame

通过`print(data.count(),len(data.columns))`打印数据的行数和列数

##### 观察分析数据

1、通过 ``data.printSchema()``打印特征类型

结果如下:

```
root
 |-- 个人编码: double (nullable = true)
 |-- 一天去两家医院的天数: integer (nullable = true)
 |-- 就诊的月数: integer (nullable = true)
 |-- 月就诊天数_MAX: integer (nullable = true)
 |-- 月就诊天数_AVG: double (nullable = true)
 |-- 月就诊医院数_MAX: integer (nullable = true)
 |-- 月就诊医院数_AVG: double (nullable = true)
 |-- 就诊次数_SUM: integer (nullable = true)
```

观察是否有字符类型的数据（本数据中不存在）



2、使用代码

```python
is_binary = data.agg(
    *[
        (F.size(F.collect_set(x)) == 2).alias(x)
        for x in data.columns
    ]
).toPandas()
is_binary.unstack()
```

提取出二元特征，将连续性特征和二元特征分开处理

结果

```
BZ_民政救助            0     True
BZ_城乡优抚            0     True
是否挂号               0     True
RES                0     True
```



3、创建四个顶级变量

```
identifiers="个人编码"     
target_column= "RES"
binary_columns=[
    "BZ_民政救助",
    "BZ_城乡优抚",
    "是否挂号"
]
continuous_columns=[
    x
    for x in data.columns
    if x not in binary_columns
    and x not in target_column
    and x not in identifiers
]
```

分别是： 包含每个记录唯一信息的列（暂时这样考虑）

​                包含我们希望预测的值的列

​                包含二分类的特征

​                包含连续的特征



4、使用代码

```python
data=data.dropna(
    how="all",
    subset=[x for x in data.columns if x not in identifiers]
)
data=data.dropna(subset=target_column)
```

删除只有null的数据



使用代码

```
data=data.fillna(0.0,subset=data.columns)
print(data.count(),len(data.columns))
```

对非全为null的数据进行0填充



5、使用代码

```python
continuous_columns=data.select(
    *[
        x for x in continuous_columns
        if (data.select(F.countDistinct(F.col(x))).collect()[0][0]!=1)
    ]
).columns
```

清楚特征值只有一个的特征，有两个特征被清除



6、导入`from pyspark.ml.feature import VectorAssembler`

使用VectorAssembler将特征列组合为单个向量列，这个转换器的用途是：通过它的transform()方法接收imputCols中的值（组合后的值），并返回一个名为outputCol的列，其中包含所有组合后的值的向量。

[VectorAssembler](https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html)

通过

```
continuous_features=VectorAssembler(
    inputCols=continuous_columns,
    outputCol="continuous_features"
)
```

对连续特征使用VectorAssembler，并返回为continuous_faetures

 

通过

```
vector_data=data.select(continuous_columns)
for x in continuous_columns:
    vector_data=vector_data.where(~F.isnull(F.col(x)))
vector_variable=continuous_features.transform(vector_data)
```

使用where方法删除非空值，因为向量列不能有null值

然后对数据进行tansform()方法

[transform()](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.transform.html)



7、导入`from pyspark.ml.stat import Correlation`

[Correlation](https://spark.apache.org/docs/3.4.2/api/python/reference/api/pyspark.ml.stat.Correlation.html)

通过

```
correlation = Correlation.corr(
    vector_variable,
    "continuous_features"
)
```

获取相关系数矩阵，采用皮尔逊相关系数



8、通过`correlation_values = correlation.collect()[0]["pearson(continuous_features)"].values`

将相关系数矩阵的相关系数放在一个列表里面，以便后续处理



9、定义方法删除高度相关的特征

```
def delHighlyCol(corr_values,inputcolumns,threshold=0.9):
    columns=inputcolumns.copy()
    # 提取高相关特征
    highly_corr = {}
    for i in range(len(columns)):
        for j in range(i):
            if abs(corr_values[i*len(columns)+j])>threshold:
                col = columns[i]
                related_col = columns[j]
                if col not in highly_corr:
                    highly_corr[col] = set()
                highly_corr[col].add(related_col)
    # 合并重复的特征并删除高相关特征
    for col,related_cols in list(highly_corr.items()):
        for related_col in list(related_cols):
            if related_col in highly_corr:
                highly_corr[col].update(highly_corr[related_col])
                del highly_corr[related_col]
    for col,related_cols in list(highly_corr.items()):
        for related_col in list(related_cols):
            columns.remove(related_col)
    return columns
```

其中threshold为阈值，相关系数绝对值大于该值的特征将不被保留



调用改方法`continuous_columnsDelHigh=delHighlyCol(corr_values=correlation_values,inputcolumns=continuous_columns,threshold=0.95)`

观察删除前后的结果，被删除了20个特征

至此，数据预处理阶段结束。接下来进入xgboost阶段



### 通过xgboost进行特征选择

1.导入必要的包

`from xgboost.spark import SparkXGBClassifier`

[xgbbost进行特征评分原理](https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/)

[xgboost在pyspark中详细文档](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.spark)



2、对4个顶级变量进行处理

```
feature_cols=continuous_columnsDelHigh+binary_columns
label_col=target_column
pred_col_name="pred"
all_cols=[identifiers]+feature_cols+[label_col]
print(all_cols)
df=data.select(
    *all_cols
).withColumnRenamed(
    "个人编码","id"
).withColumn("id",F.monotonically_increasing_id()+1) #更新编号
df.show()
```



3、对数据使用VectorAssembler和tansform

```
vec_assembler=VectorAssembler(inputCols=feature_cols,outputCol="features")
df_input = vec_assembler.transform(df)
```



4、构建xgboost模型

```
classifier=SparkXGBClassifier(
    features_col="features",
    label_col=label_col,
    num_workers=3,
)
```

此处构建较为简单，后续可以优化



5、进行模型训练，并指定特征名（xgboost在训练过程中会将特征名简化为f1,f2,f3.......)

```
model = classifier.fit(df_input)
model.get_booster().feature_names = feature_cols
```



6、使用`get_feature_importances`函数

`features_importance = model.get_feature_importances(importance_type="total_gain")`

[get_feature_importances](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.spark)

[get_feature_importances函数中importance_type的不同参数对应的含义](https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7)



7、按照得total_gain分进行降序排列，并将结果转换为字典

```
features_importance = sorted(
    features_importance.items(),
    key=lambda x:x[1],
    reverse=True
)
features_importance = dict(features_importance)
```



8、导入二分类评价包

`from pyspark.ml.evaluation import BinaryClassificationEvaluator`

[BinaryClassificationEvaluator](https://spark.apache.org/docs/3.4.2/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html)



9、定义特征选择方法

```
kept = []
i=1
rem_dict = {}
for feature in features_importance.keys():
    kept.append(feature)
    df_kept = df.select(*kept,label_col)
    trainDF,testDF = df_kept.randomSplit([0.8,0.2],seed=14)
    kept_assembler = VectorAssembler(inputCols=kept,outputCol="feature_subset")
    classifierInFor = SparkXGBClassifier(
        max_depth = 5,
        missing = 0,
        features_col="feature_subset",
        label_col=label_col,
        num_workers=3,
        raw_prediction_col="raw_prediction"
    )
    with_selected_feature = kept_assembler.transform(trainDF)
    test_data = kept_assembler.transform(testDF)
    xgb_model = classifierInFor.fit(with_selected_feature)
    pre_xgb_df = xgb_model.transform(test_data)
    evaluator_model = BinaryClassificationEvaluator(
        rawPredictionCol="raw_prediction",
        labelCol=label_col,
        metricName="areaUnderROC"
    )
    eva_val = evaluator_model.evaluate(pre_xgb_df)
    rem_dict[i] = eva_val
    i=i+1
```

该方法的主要思路是：对降序排列后的特征，进行遍历。按照遍历次数，第n次遍历取前n个特征。并将所取特征作为筛选后的特征组，放入到xgboost中进行训练，在通过二分类评价器进行评价得到改特征组下的AUC。最后观察AUC随特征数量的变化情况，来进行特征选择。

例如在最后结果中：

```
{1: 0.7935234933562302,  只选第一个特征
 2: 0.820094107582077,   只选前两个特征
 3: 0.9065800449149825,  只选前三个特征
 4: 0.9002063950379668,  以此类推
 5: 0.9120692974013482,
 6: 0.9158699604320408,
 7: 0.9249353010373242,     观察到在选前七个特征后，AUC的变化并不明显，故可以考虑将前七个特征作为筛选结果
 8: 0.9232787937119049,
 9: 0.9311923858410878,
 10: 0.9208095390867314,
 11: 0.9234456207892212,
 12: 0.9146187573521553,
 13: 0.9158763768580931,
 14: 0.9206844187787415,
 15: 0.9156325526681646,
 16: 0.926544754571704,
 17: 0.9208213025344889,
 18: 0.9162528071864,
 19: 0.9295155598331756,
 20: 0.9183360068441854,
 21: 0.9210865148112566,
 22: 0.9248508180943245,
 23: 0.9204063736498784,
 24: 0.9241450112287467,
 25: 0.9260592450005358,
 26: 0.9280269489894132,
 27: 0.91366057106192,
 28: 0.9173799593626363,
 29: 0.922470324029517,
 30: 0.922495989733719,
 31: 0.9199828895305325,
 32: 0.915681745267891,
 33: 0.9157309378676105,
 34: 0.915822906641005,
 35: 0.9225772644636936,
 36: 0.9185691369906971,
 37: 0.9236509464228397,
 38: 0.9193883007165013,
 39: 0.9244957758528501,
 40: 0.9166570420275891,
 41: 0.912905571596619,
 42: 0.9207058068655787,
 43: 0.919407549994656,
 44: 0.9104737461234087,
 45: 0.9214351406266685,
 46: 0.9143984600577494,
 47: 0.9160795636830272,
 48: 0.9239268527430236}
```

本代码到此结束