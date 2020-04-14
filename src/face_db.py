# 读取数据库中的人脸数据
import pymysql as mysql


class FaceDB:
    def __init__(self):
        self.db = mysql.connect(host="localhost", user="root",
                                passwd="123456", database="face")
        self.cursor = self.db.cursor()

    def get_all_faces(self):
        self.cursor.execute("SELECT name, face_id FROM face")
        result = self.cursor.fetchall()
        return result

    def insert_face(self, name, face_id):
        sql = "INSERT INTO face (name, face_id) VALUES (%s, %s)"
        val = (name, face_id)
        self.cursor.execute(sql, val)
        self.db.commit()


if __name__ == '__main__':
    db = FaceDB()
    for i in db.get_all_faces():
        print(i)
