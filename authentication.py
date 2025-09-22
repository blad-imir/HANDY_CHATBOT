import os
from flask import Flask, redirect, url_for, render_template, session
from flask_dance.contrib.google import make_google_blueprint, google

app = Flask(__name__)

# === Security ===
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecret")

# === Google OAuth Setup ===
app.config["GOOGLE_OAUTH_CLIENT_ID"] = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "1013140181940-i5epgmskq30f60kp9qvamd8vosddubei.apps.googleusercontent.com"
)
app.config["GOOGLE_OAUTH_CLIENT_SECRET"] = os.environ.get(
    "GOOGLE_CLIENT_SECRET",
    "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"
)

google_bp = make_google_blueprint(
    client_id=app.config["GOOGLE_OAUTH_CLIENT_ID"],
    client_secret=app.config["GOOGLE_OAUTH_CLIENT_SECRET"],
    scope=["profile", "email"],
    redirect_to="dashboard"  # after login, go here
)
app.register_blueprint(google_bp, url_prefix="/login")

# === Routes ===
@app.route("/")
def home():
    # Show login page if user not logged in
    if not google.authorized:
        return render_template("login.html")
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    # Require login
    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        return "❌ Failed to fetch user info", 400

    user_info = resp.json()
    session["user"] = user_info

    # ✅ Instead of showing dashboard.html, load chatbot UI
    return render_template("index.html", user=user_info)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
