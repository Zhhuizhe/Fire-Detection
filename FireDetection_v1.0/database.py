import pyodbc
import datetime
import pandas as pd


class Database(object):
    def __init__(self, server, database, username, password):
        self.__tablename_dict = {"fire_output": 1, "relative_fire": 2, "absolute_fire": 3}
        self.__conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
        if not self.__conn:
            print("数据库连接失败")
        else:
            print("已成功连接数据库" + database)
        self.__cursor = self.__conn.cursor()

    def df_insert(self, dataframe, tablename):
        table_no = self.__tablename_dict.get(tablename)
        if not table_no:
            print("ERROR:can't find table " + tablename)
            return
        if table_no == self.__tablename_dict['fire_output']:
            for index, row in dataframe.iterrows():
                self.__cursor.execute("""INSERT INTO fire_output(
                中心经度,中心纬度,日期,时间,亚像元火点面积,林地概率,草地概率,
                农田概率,其他概率,[7波段像元值],[14波段像元值],所在市,所在县,备注) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row[2], row[3], row[0], row[1], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13])
                self.__cursor.commit()
        elif table_no == self.__tablename_dict['relative_fire']:
            for index, row in dataframe.iterrows():
                self.__cursor.execute("""INSERT INTO relative_fire (
                中心经度,中心纬度,日期,时间,亚像元火点面积,林地概率,草地概率,
                农田概率,其他概率,7波段像元值,14波段像元值,备注) values(?,?,?,?,?,?,?,?,?,?,?,?)""", row[2], row[3], row[0], row[1], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11])
                self.__cursor.commit()
        elif table_no == self.__tablename_dict['absolute_fire']:
            for index, row in dataframe.iterrows():
                self.__cursor.execute("""INSERT INTO relative_fire (
                中心经度,中心纬度,日期,时间,亚像元火点面积,林地概率,草地概率,
                农田概率,其他概率,7波段像元值,14波段像元值,备注) values(?,?,?,?,?,?,?,?,?,?,?,?)""", row[2], row[3], row[0], row[1], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11])
                self.__cursor.commit()

    def img_insert(self, img_path, filename):
        date = datetime.datetime.strptime(filename[4:12], "%Y%m%d")
        time = datetime.datetime.strptime(filename[12:16], "%H%M")
        self.__cursor.execute("""INSERT INTO Image (日期,时间,真彩图,火情图,云图,[7通道图像],[7_14通道图像],全通道图像) values(?,?,?,?,?,?,?,?)""",
                              date, time, img_path[0], img_path[1], img_path[2], img_path[3], img_path[4], img_path[5])
        self.__cursor.commit()

    def select_query(self, script):
        rows = self.__cursor.execute(script).fetchall()
        self.__conn.commit()
        return rows

    def close(self):
        self.__cursor.close()
        self.__conn.close()


if __name__ == '__main__':
    df = pd.read_csv("E:/山火项目/final_output/SHJC2020120214305.csv", encoding="gbk", header=0)
    test = Database(r"DESKTOP-52LI12S\ZHZSQL", "fire_detection", "sa", "91757@k-a")
    test.close()
    test.df_insert(df)
