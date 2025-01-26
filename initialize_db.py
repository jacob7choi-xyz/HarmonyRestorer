from app import app, db  # Import your app and db object

# Initialize the database by creating tables
with app.app_context():
    db.create_all()  # This creates all tables defined in your models
