import polars as pl
g = pl.DataFrame({"type_CASH_IN" :1, "type_CASH_OUT" :0, "type_PAYMENT" :0, "type_TRANSFER" :0, "amount"   :6000.56, "type_2_CC" :1, "type_2_CM" :0,
                   "day" :1, "part_of_the_day_madrugada" :1, "part_of_the_day_mañana" :0, "part_of_the_day_noche" :0, "part_of_the_day_tarde": 0})
print (g)
