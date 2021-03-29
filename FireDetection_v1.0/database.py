import pyodbc
import datetime
import pandas as pd


class Database(object):
    def __init__(self, server, database, username, password):
        self.__conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
        if not self.__conn:
            print("数据库连接失败")
        else:
            print("已成功连接数据库" + database)
        self.__cursor = self.__conn.__cursor()

    def df_insert(self, dataframe, time, tablename):
        date = datetime.datetime.strptime("20201202", "%Y%m%d")
        time = datetime.datetime.strptime("1430", "%H%M")
        for index, row in dataframe.iterrows():
            # 将dataframe插入至表格relative_fire中
            # row[1]--中心经度, row[2]--中心纬度, row[3]--热点像元个数, row[4]--热点面积, row[5]--林地概率
            # row[6]--草地概率, row[7]--农田概率, row[8]--其他概率, row[9]--平均置信度, row[10]--备注
            self.__cursor.execute("""INSERT INTO relative_fire (
            中心经度,中心纬度,日期,
            时间,热点像元个数,热点面积,
            林地概率,草地概率,农田概率,
            其他概率,平均可信度,备注) values(?,?,?,?,?,?,?,?,?,?,?,?)""", row[1], row[2], date, time, row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10])
            self.__cursor.commit()

    def execute_script(self, script):
        self.__cursor.execute(script)
        self.__conn.commit()

    def close(self):
        self.__cursor.close()
        self.__conn.close()


if __name__ == '__main__':
    df = pd.read_csv("E:/山火项目/final_output/SHJC2020120214305.csv", encoding="gbk", header=0)
    test = Database(r"DESKTOP-52LI12S\ZHZSQL", "fire_detection", "sa", "91757@k-a")
    test.close()
    test.df_insert(df)
