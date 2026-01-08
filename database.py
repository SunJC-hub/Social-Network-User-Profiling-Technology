import sqlite3

def init_emotion_icon_db():
    conn = sqlite3.connect("emotion_icons.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotions (
            emotion_name TEXT PRIMARY KEY,
            icon TEXT NOT NULL
        )
    ''')
    data = [
        ("happiness", "😄"),
        ("sadness", "😭"),
        ("anger", "😠"),
        ("fear", "😨"),
        ("disgust", "🤢"),
        ("surprise", "😲"),
        ("like", "❤️"),
        ("none", "😐")
    ]
    cursor.executemany("INSERT OR REPLACE INTO emotions (emotion_name, icon) VALUES (?, ?)", data)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_emotion_icon_db()
    print("情感颜文字数据库已初始化完毕。")
