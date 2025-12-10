// MongoDB initialization script
db = db.getSiblingDB('econsultation');

// Create collections
db.createCollection('users');
db.createCollection('comments');
db.createCollection('analysis_results');
db.createCollection('reports');
db.createCollection('sessions');

// Create indexes for better performance
db.users.createIndex({ "email": 1 }, { unique: true });
db.users.createIndex({ "username": 1 }, { unique: true });
db.comments.createIndex({ "created_at": -1 });
db.comments.createIndex({ "sentiment": 1 });
db.analysis_results.createIndex({ "timestamp": -1 });
db.sessions.createIndex({ "created_at": 1 }, { expireAfterSeconds: 86400 });

// Create default admin user
db.users.insertOne({
    username: "admin",
    email: "admin@econsultation.gov",
    full_name: "System Administrator",
    role: "admin",
    hashed_password: "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5lW7QK5oCqH6W", // password: admin123
    is_active: true,
    created_at: new Date(),
    updated_at: new Date()
});

print('Database initialized successfully!');
