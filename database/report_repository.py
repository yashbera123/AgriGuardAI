from database.database import Database


class ReportRepository:
    """Repository for managing saved reports (future expansion)."""

    def __init__(self):
        self.db = Database()
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create the reports table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS reports (
            id INT AUTO_INCREMENT PRIMARY KEY,
            prediction_id INT,
            report_id_string VARCHAR(100) NOT NULL,
            generated_at DATETIME NOT NULL,
            pdf_path VARCHAR(255),
            FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE SET NULL
        )
        """
        self.db.execute_query(query)

    def save_report(
        self,
        prediction_id: int,
        report_id_string: str,
        generated_at: str,
        pdf_path: str = None,
    ) -> int:
        """Save a new report record."""
        query = """
        INSERT INTO reports (prediction_id, report_id_string, generated_at, pdf_path)
        VALUES (%s, %s, %s, %s)
        """
        params = (prediction_id, report_id_string, generated_at, pdf_path)
        return self.db.execute_query(query, params)

    def get_report(self, report_id: int) -> dict:
        """Retrieve a specific report by ID."""
        query = "SELECT * FROM reports WHERE id = %s"
        results = self.db.execute_query(query, (report_id,), fetch=True)
        return results[0] if results else None
