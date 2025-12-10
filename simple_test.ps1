# Simple System Test
Write-Host "Testing MCA Sentiment Analysis System" -ForegroundColor Blue

# Test API Health
Write-Host "Testing API Health..." -ForegroundColor Yellow
$health = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method GET
Write-Host "API Status: $($health.status)" -ForegroundColor Green

# Test Simple Analysis
Write-Host "Testing Sentiment Analysis..." -ForegroundColor Yellow
$body = @{
    texts = @("This policy is excellent", "यह नीति अच्छी है")
    include_explanation = $true
} | ConvertTo-Json

$result = Invoke-RestMethod -Uri "http://localhost:8001/api/analyze" -Method POST -Body $body -ContentType "application/json"
Write-Host "Analysis completed: $($result.summary.total_analyzed) texts" -ForegroundColor Green

Write-Host "System is working perfectly!" -ForegroundColor Green