"""
Quick test to verify the dashboard fix works with different column names
"""
import pandas as pd

# Create test data with different column name (not 'comment')
test_data = pd.DataFrame({
    'feedback': [
        'I strongly support this initiative as it will greatly benefit our community',
        'This policy lacks clarity and may cause confusion among stakeholders',
        'I have mixed feelings about this approach - some good points but concerns too',
        'Excellent proposal that addresses key issues effectively',
        'The implementation timeline seems unrealistic and problematic'
    ],
    'stakeholder_id': [1, 2, 3, 4, 5],
    'date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05']
})

# Save test file
test_data.to_csv('test_feedback_data.csv', index=False)
print("✅ Test data created with 'feedback' column (not 'comment')")
print("📊 Sample data:")
print(test_data.head())
print("\n🎯 This will test if the dashboard can handle different column names")
print("💾 File saved as: test_feedback_data.csv")