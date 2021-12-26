import numpy as np
import pandas as pd

###############################################
# Veri setini inceleyelim.
###############################################

##: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("datasets/persona.csv")

def check_df(dataframe, head=5, tail=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(tail))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    print("##################### Describe #####################")
    print(dataframe.describe().T)

    print("##################### Info #####################")
    print(dataframe.info())


check_df(df)

## Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

## Kaç unique PRICE vardır?
df["PRICE"].nunique()
df["PRICE"].unique()

## Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

## angi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()
df.groupby(["COUNTRY"])["PRICE"].count().reset_index()

## Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby(["COUNTRY"])["PRICE"].sum().reset_index()
#: OR
df.groupby(["COUNTRY"]).agg({"PRICE": "sum"}).reset_index()

## SOURCE türlerine göre göre satış sayıları nedir?
df["SOURCE"].value_counts()

## Ülkelere göre PRICE ortalamaları nedir?
df.groupby(["COUNTRY"])["PRICE"].mean().reset_index()
#: OR
df.groupby(["COUNTRY"]).agg({"PRICE": "mean"}).reset_index()


## SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby(["SOURCE"])["PRICE"].mean().reset_index()
#: OR
df.groupby(["SOURCE"]).agg({"PRICE": "mean"}).reset_index()

## COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean().reset_index()
#: OR
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"}).reset_index()

###############################################
# COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
###############################################
col_list = ["COUNTRY", "SOURCE", "SEX", "AGE"]
df.groupby(col_list)["PRICE"].mean()
df.groupby(col_list).agg({"PRICE": "mean"})

###############################################
# Çıktıyı PRICE’a göre sıralayınız.
###############################################
#: Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
#: Çıktıyıagg_df olarak kaydediniz.

col_list = ["COUNTRY", "SOURCE", "SEX", "AGE"]
agg_df = df.groupby(col_list).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

###############################################
# Index’te yer alan isimleri değişken ismine çeviriniz.
###############################################
#: Üçüncü sorunun çıktısında yer alan price dışındaki tüm değişkenler index isimleridir.
#: Bu isimleri değişken isimlerine çeviriniz.

agg_df = agg_df.reset_index()
#: OR
agg_df.reset_index(inplace=True)

###############################################
# age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
###############################################
#: Age sayısal değişkenini kategorik değişkene çeviriniz.
#: Aralıkları ikna edici şekilde oluşturunuz.
#: Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'

def age_cut(x):
    if 0 < x["AGE"] <= 18:
        return "0_18"
    elif x["AGE"] <= 23:
        return "19_23"
    elif x["AGE"] <= 30:
        return "24_30"
    elif x["AGE"] <= 40:
        return "31_40"
    elif x["AGE"] <= 66:
        return "41_66"
    else:
        return "none"


agg_df["AGE_CAT"] = agg_df.apply(lambda x: age_cut(x), axis=1)

# OR
bin_list = [0, 18, 23, 30, 40, 66]
label_list = ["0_18", "19_23", "24_30", "31_40", "41_66"]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=bin_list, labels=label_list)

###############################################
# Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
###############################################
#: Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz. 
#: Yeni eklenecek değişkenin adı: customers_level_based
#: Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.

def create_customers_level_based(x):
    return "_".join([x["COUNTRY"], x["SOURCE"], x["SEX"], x["AGE_CAT"]]).upper()


agg_df["customers_level_based"] = agg_df.apply(create_customers_level_based, axis=1)
agg_df[["customers_level_based", "PRICE"]]
agg_df = agg_df.groupby(["customers_level_based"]).agg({"PRICE": "mean"}).reset_index()


###############################################
# Yeni müşterileri (personaları) segmentlere ayırınız.
###############################################
#: Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
#: Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
#: Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız). 
#: C segmentini analiz ediniz (Veri setinden sadece C segmentini çekip analiz ediniz).


agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

segment_analiz = agg_df.groupby(["SEGMENT"]).agg({"PRICE": ["min", "mean", "max", "sum", np.median]}).reset_index()
segment_c = segment_analiz[segment_analiz["SEGMENT"] == "C"]


###############################################
# Yeni gelen müşterileri segmentlerine göre sınıflandırınız ve ne kadar gelir getirebileceğini tahmin ediniz.
###############################################

def customer_mean_price(df: pd.DataFrame, new_user: str) -> float:
    return float(df[df["customers_level_based"] == new_user]["PRICE"])

#: 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user_tur = "TUR_ANDROID_FEMALE_31_40"
new_user_mean_price_tur = customer_mean_price(df=agg_df, new_user=new_user_tur)
print("TUR:", new_user_mean_price_tur)

#: 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
new_user_far = "FRA_IOS_FEMALE_31_40"
new_user_mean_price_far = customer_mean_price(df=agg_df, new_user=new_user_far)
print("FRA:", new_user_mean_price_far)

