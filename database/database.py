import mysql.connector
from mysql.connector import Error

from config import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER
from utils.logger_config import get_logger

logger = get_logger(__name__)


class Database:
    """Manages MySQL database connections for the application."""

    def __init__(self):
        """Initialize database connection parameters from config."""
        self.host = DB_HOST
        self.user = DB_USER
        self.password = DB_PASSWORD
        self.database = DB_NAME

    def get_connection(self):
        """Create and return a new MySQL connection."""
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            if connection.is_connected():
                return connection
        except Error as e:
            logger.error(f"Error connecting to MySQL Database: {e}")
            from utils.exceptions import DatabaseError

            raise DatabaseError(f"Database connection failed: {e}")
        return None

    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Execute a query, committing changes or fetching results."""
        connection = self.get_connection()
        if not connection:
            return None

        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params or ())

            if fetch:
                result = cursor.fetchall()
                return result
            else:
                connection.commit()
                return cursor.lastrowid
        except Error as e:
            logger.error(f"Database query error: {e}")
            from utils.exceptions import DatabaseError

            raise DatabaseError(f"Database query failed: {e}")
        finally:
            if cursor:
                cursor.close()
            if connection.is_connected():
                connection.close()
