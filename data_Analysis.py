import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("hotel_bookings.csv")

# Convert date columns
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

# Analytics: Cancellation Rate
cancellation_rate = (df['is_canceled'].mean()) * 100
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

# Analytics: Revenue Trends Over Time
df['revenue'] = df['adr'] * df['stays_in_week_nights']  # Estimated Revenue
revenue_trends = df.groupby(df['reservation_status_date'].dt.to_period('M'))['revenue'].sum()

# Plot Revenue Trends
plt.figure(figsize=(10, 5))
revenue_trends.plot(kind='line', marker='o', title="Monthly Revenue Trends")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

# Booking Lead Time Distribution
sns.histplot(df['lead_time'], bins=50, kde=True)
plt.title("Lead Time Distribution")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Frequency")
plt.show()
