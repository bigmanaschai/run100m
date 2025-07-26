# database.py - Database initialization and operations

import sqlite3
import pandas as pd
from datetime import datetime
from config import Config


class Database:
    """Database operations for the running performance app"""

    def __init__(self):
        self.db_name = Config.DATABASE_NAME
        self.init_db()

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_name)

    def init_db(self):
        """Initialize database with required tables"""
        conn = self.get_connection()
        c = conn.cursor()

        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users
        (
            id
            INTEGER
            PRIMARY
            KEY
            AUTOINCREMENT,
            username
            TEXT
            UNIQUE
            NOT
            NULL,
            password
            TEXT
            NOT
            NULL,
            user_type
            TEXT
            NOT
            NULL,
            coach_id
            INTEGER,
            created_at
            TIMESTAMP
            DEFAULT
            CURRENT_TIMESTAMP,
            FOREIGN
            KEY
                     (
            coach_id
                     ) REFERENCES users
                     (
                         id
                     ))''')

        # Performance data table
        c.execute('''CREATE TABLE IF NOT EXISTS performance_data
        (
            id
            INTEGER
            PRIMARY
            KEY
            AUTOINCREMENT,
            runner_id
            INTEGER
            NOT
            NULL,
            coach_id
            INTEGER,
            test_date
            TIMESTAMP
            DEFAULT
            CURRENT_TIMESTAMP,
            range_0_25_data
            TEXT,
            range_25_50_data
            TEXT,
            range_50_75_data
            TEXT,
            range_75_100_data
            TEXT,
            video_0_25_path
            TEXT,
            video_25_50_path
            TEXT,
            video_50_75_path
            TEXT,
            video_75_100_path
            TEXT,
            max_speed
            REAL,
            avg_speed
            REAL,
            total_time
            REAL,
            notes
            TEXT,
            FOREIGN
            KEY
                     (
            runner_id
                     ) REFERENCES users
                     (
                         id
                     ),
            FOREIGN KEY
                     (
                         coach_id
                     ) REFERENCES users
                     (
                         id
                     ))''')

        # Performance metrics table (for detailed analytics)
        c.execute('''CREATE TABLE IF NOT EXISTS performance_metrics
        (
            id
            INTEGER
            PRIMARY
            KEY
            AUTOINCREMENT,
            performance_id
            INTEGER
            NOT
            NULL,
            metric_name
            TEXT
            NOT
            NULL,
            metric_value
            REAL
            NOT
            NULL,
            created_at
            TIMESTAMP
            DEFAULT
            CURRENT_TIMESTAMP,
            FOREIGN
            KEY
                     (
            performance_id
                     ) REFERENCES performance_data
                     (
                         id
                     ))''')

        conn.commit()
        conn.close()

    def get_user(self, username, password_hash):
        """Get user by username and password"""
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("SELECT id, user_type, coach_id FROM users WHERE username=? AND password=?",
                  (username, password_hash))
        result = c.fetchone()
        conn.close()
        return result

    def create_user(self, username, password_hash, user_type, coach_id=None):
        """Create new user"""
        conn = self.get_connection()
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, user_type, coach_id) VALUES (?, ?, ?, ?)",
                      (username, password_hash, user_type, coach_id))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False

    def get_coaches(self):
        """Get all coaches"""
        conn = self.get_connection()
        df = pd.read_sql_query("SELECT id, username FROM users WHERE user_type='coach'", conn)
        conn.close()
        return df

    def get_runners_for_coach(self, coach_id):
        """Get all runners for a specific coach"""
        conn = self.get_connection()
        df = pd.read_sql_query(
            "SELECT id, username FROM users WHERE user_type='runner' AND coach_id=?",
            conn, params=(coach_id,)
        )
        conn.close()
        return df

    def get_all_runners(self):
        """Get all runners"""
        conn = self.get_connection()
        df = pd.read_sql_query("SELECT id, username FROM users WHERE user_type='runner'", conn)
        conn.close()
        return df

    def save_performance_data(self, data):
        """Save performance data to database"""
        conn = self.get_connection()
        c = conn.cursor()

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])

        c.execute(f"INSERT INTO performance_data ({columns}) VALUES ({placeholders})",
                  list(data.values()))

        performance_id = c.lastrowid
        conn.commit()
        conn.close()

        return performance_id

    def get_performance_data(self, user_id, user_type):
        """Get performance data based on user type and ID"""
        conn = self.get_connection()

        if user_type == 'admin':
            query = """
                    SELECT p.*, u.username as runner_name, c.username as coach_name
                    FROM performance_data p
                             JOIN users u ON p.runner_id = u.id
                             LEFT JOIN users c ON p.coach_id = c.id
                    ORDER BY p.test_date DESC \
                    """
            df = pd.read_sql_query(query, conn)
        elif user_type == 'coach':
            query = """
                    SELECT p.*, u.username as runner_name
                    FROM performance_data p
                             JOIN users u ON p.runner_id = u.id
                    WHERE p.coach_id = ?
                    ORDER BY p.test_date DESC \
                    """
            df = pd.read_sql_query(query, conn, params=(user_id,))
        else:  # runner
            query = """
                    SELECT p.*, c.username as coach_name
                    FROM performance_data p
                             LEFT JOIN users c ON p.coach_id = c.id
                    WHERE p.runner_id = ?
                    ORDER BY p.test_date DESC \
                    """
            df = pd.read_sql_query(query, conn, params=(user_id,))

        conn.close()
        return df

    def get_all_users(self):
        """Get all users with their details"""
        conn = self.get_connection()
        df = pd.read_sql_query("""
                               SELECT u1.id, u1.username, u1.user_type, u2.username as coach_name, u1.created_at
                               FROM users u1
                                        LEFT JOIN users u2 ON u1.coach_id = u2.id
                               ORDER BY u1.created_at DESC
                               """, conn)
        conn.close()
        return df

    def delete_user(self, user_id):
        """Delete user by ID"""
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        conn.close()


# Create global database instance
db = Database()