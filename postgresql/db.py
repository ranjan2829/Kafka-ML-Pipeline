from fastapi import FastAPI
import psycopg2

app=FastAPI()
conn=psycopg2.connect(
    database="app",
    user="postgres",
    password="1234",
    host="localhost",
    port="8090"
)
cur=conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name TEXT,
    email TEXT
)
""")

conn.commit()
@app.post("/add")
def add_user():
    cur.execute("INSERT INTO users (name,email) VALUES ('ranjan','')")
    user_id = cursor.fetchone()[0]
    conn.commit()
    return {"id": user_id, "name": name, "email": email}

@app.get("/user/{user_id}")
def get_user(user_id:int):
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    if user:
        return {"id": user[0], "name": user[1], "email": user[2]}
    else:
        return {"error": "User not found"}