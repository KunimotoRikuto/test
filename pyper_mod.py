#author;R.Kunimoto, TAKENAKA co.
#coding:utf-8
import subprocess
import pyper

r=pyper.R()
end =r("""
library(h2o)
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)

speci = "const"
month <- 1
dim <- 3

h2o_tr <- h2o.importFile(path = paste(c("C:\\Users\\1500570\\Documents\\R\\WS\\dataset\\",speci,"\\logit_train_",dim,".csv"),collapse=""))
h2o_ts <- h2o.importFile(path = paste(c("C:\\Users\\1500570\\Documents\\R\\WS\\dataset\\",speci,"\\logit_test_",dim,".csv"),collapse=""))

df_tr1 <- as.data.frame(h2o_tr)
df_ts <- as.data.frame(h2o_ts)

training = as.h2o(df_tr1)
testing = as.h2o(df_ts)

dl <- h2o.deeplearning(x=1:dim*10, y=dim*10+month, training_frame=training)
summarying <- h2o.predict(dl, newdata = testing)
""")

print(r("dl"))