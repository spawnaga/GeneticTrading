
import requests
import time

def test_dashboard():
    """Test if the dashboard is accessible."""
    try:
        response = requests.get("http://0.0.0.0:5000/trading_dashboard.html", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is accessible!")
            print(f"Response length: {len(response.text)} characters")
        else:
            print(f"❌ Dashboard returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Connection refused - Dashboard server not running")
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Testing dashboard connectivity...")
    test_dashboard()
