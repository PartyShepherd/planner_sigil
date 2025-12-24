from flask import Flask, render_template, request, send_file
from datetime import datetime, timedelta
import pytz
import os
import requests
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------- Flask App ----------------
app = Flask(__name__)

# ---------------- Planner Settings ----------------
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
LOCATION = "Pittsburgh, US"
LOG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)
RITUALS = ["LIRP", "RR", "LBRP", "LIRH", "MP", "GIRP", "GIRH", "RC"]

# ---------------- Planner Helper Functions ----------------

def get_elemental_quarter():
    current_hour = datetime.now(pytz.timezone("US/Eastern")).hour
    if 0 <= current_hour < 6:
        return "Earth (12 AM - 6 AM)"
    elif 6 <= current_hour < 12:
        return "Air (6 AM - 12 PM)"
    elif 12 <= current_hour < 18:
        return "Fire (12 PM - 6 PM)"
    else:
        return "Water (6 PM - 12 AM)"

def get_moon_phase():
    try:
        ref_date = datetime(2000, 1, 6, 12, 24, tzinfo=pytz.utc)
        current_date = datetime.now(pytz.timezone("US/Eastern")).astimezone(pytz.utc)
        days_since_ref = (current_date - ref_date).days + (current_date - ref_date).seconds / 86400.0
        lunation = 29.53058867
        phase = (days_since_ref % lunation) / lunation
        if phase < 0.03 or phase > 0.97:
            return "New Moon"
        elif phase < 0.25:
            return "Waxing Crescent"
        elif phase < 0.27:
            return "First Quarter"
        elif phase < 0.50:
            return "Waxing Gibbous"
        elif phase < 0.53:
            return "Full Moon"
        elif phase < 0.75:
            return "Waning Gibbous"
        elif phase < 0.77:
            return "Last Quarter"
        else:
            return "Waning Crescent"
    except Exception:
        return "Moon phase data unavailable"

def get_weather():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={LOCATION}&appid={OPENWEATHER_API_KEY}&units=metric"
        data = requests.get(url).json()
        if "main" in data:
            temp_c = data["main"]["temp"]
            temp_f = (temp_c * 9 / 5) + 32
            condition = data["weather"][0]["description"]
            return f"{temp_f:.1f}°F, {condition.capitalize()}"
        else:
            return "Weather data unavailable"
    except Exception:
        return "Weather data unavailable"

def get_sunrise_sunset():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={LOCATION}&appid={OPENWEATHER_API_KEY}&units=metric"
        data = requests.get(url).json()
        if "sys" in data:
            timezone_offset = data["timezone"]
            sunrise = datetime.utcfromtimestamp(data["sys"]["sunrise"] + timezone_offset).replace(
                tzinfo=pytz.utc).astimezone(pytz.timezone("US/Eastern"))
            sunset = datetime.utcfromtimestamp(data["sys"]["sunset"] + timezone_offset).replace(
                tzinfo=pytz.utc).astimezone(pytz.timezone("US/Eastern"))
            return sunrise, sunset
        else:
            return None, None
    except Exception:
        return None, None

def get_planetary_hour():
    try:
        sunrise, sunset = get_sunrise_sunset()
        if not sunrise or not sunset:
            return "Planetary hour data unavailable"
        planets = ["Sun", "Venus", "Mercury", "Moon", "Saturn", "Jupiter", "Mars"]
        start_planet = {
            "Sunday": "Sun", "Monday": "Moon", "Tuesday": "Mars",
            "Wednesday": "Mercury", "Thursday": "Jupiter",
            "Friday": "Venus", "Saturday": "Saturn"
        }
        day_of_week = datetime.now(pytz.timezone("US/Eastern")).strftime("%A")
        day_start_planet = start_planet[day_of_week]
        daytime_duration = (sunset - sunrise) / 12
        nighttime_duration = (sunrise + timedelta(days=1) - sunset) / 12
        current_time = datetime.now(pytz.timezone("US/Eastern"))
        if sunrise <= current_time < sunset:
            hours_since_sunrise = int((current_time - sunrise) / daytime_duration)
            current_planet = planets[(planets.index(day_start_planet) + hours_since_sunrise) % 7]
        else:
            hours_since_sunset = int((current_time - sunset) / nighttime_duration)
            current_planet = planets[(planets.index(day_start_planet) + 12 + hours_since_sunset) % 7]
        return current_planet
    except Exception:
        return "Planetary hour data unavailable"

# ---------------- Sigil Generator Setup ----------------
letter_mapping = {
    "A": [("A", "#FFFF00")], "E": [("A", "#FFFF00")], "B": [("B", "#FFFF00")],
    "C": [("G", "#0000FF")], "CH": [("Ch", "#FFD700")], "H": [("H", "#FF0000")],
    "G": [("G", "#0000FF")], "GH": [("GH", "#0000FF")], "D": [("D", "#008000")],
    "DH": [("DH", "#008000")], "V": [("V", "#FF4500")], "U": [("V", "#FF4500")],
    "W": [("V", "#FF4500")], "O": [("O", "#4B0082")], "Z": [("Z", "#FFA500")],
    "T": [("T", "#FFFF00")], "TH": [("TH", "#4B0082")], "I": [("I", "#9ACD32")],
    "J": [("I", "#9ACD32")], "Y": [("I", "#9ACD32")], "K": [("K", "#800080")],
    "KH": [("KH", "#9400D3")], "L": [("L", "#008000")], "M": [("M", "#0000FF")],
    "N": [("N", "#20B2AA")], "S": [("S", "#FF4500")], "SH": [("Sh", "#FF0000")],
    "P": [("P", "#FF0000")], "PH": [("PH", "#FFA500")], "F": [("F", "#FF0000")],
    "X": [("X", "#800080")], "TZ": [("X", "#800080")], "Q": [("Q", "#9400D3")],
    "R": [("R", "#FFA500")], "RH": [("RH", "#FFD700")],
}

def preprocess_word(word):
    word = word.upper()
    processed_word = []
    skip_next = False
    for i in range(len(word)):
        if skip_next:
            skip_next = False
            continue
        if i < len(word) - 1:
            two_letter_combo = word[i] + word[i + 1]
            if two_letter_combo in letter_mapping:
                processed_word.append(two_letter_combo)
                skip_next = True
                continue
        if word[i] in letter_mapping:
            processed_word.append(word[i])
    return processed_word

def draw_rose_sigil(word="JAMES"):
    word = preprocess_word(word)
    all_positions = {}
    ordered_outer_letters = ["H","Z","V","E","Q","X","O","S","N","L","I","T","Ch"]
    ordered_middle_letters = ["R","RH","P","PH","F","K","KH","TH","G","GH","D","DH","B"]
    ordered_mother_letters = ["M","A","Sh"]
    all_positions.update({
        letter: (-np.cos(i*(2*np.pi/len(ordered_outer_letters)))*2,
                 np.sin(i*(2*np.pi/len(ordered_outer_letters)))*2)
        for i, letter in enumerate(ordered_outer_letters)
    })
    all_positions.update({
        letter: (-np.cos(i*(2*np.pi/len(ordered_middle_letters)))*1.2,
                 np.sin(i*(2*np.pi/len(ordered_middle_letters)))*1.2)
        for i, letter in enumerate(ordered_middle_letters)
    })
    all_positions["A"] = (0,0.7)
    all_positions["M"] = (-0.7,0)
    all_positions["Sh"] = (0.7,0)
    fig, ax = plt.subplots(figsize=(6,6), facecolor='#D3D3D3')
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_aspect('equal')
    ax.add_patch(Circle((0,0),2.5, fill=False, edgecolor='black', lw=2))
    prev_pos = prev_color = first_pos = last_pos = None
    seen_positions = {}
    for char in word:
        if char in letter_mapping:
            letter, color = letter_mapping[char][0]
            if letter not in all_positions:
                continue
            x, y = all_positions[letter]
            ax.text(x, y, letter, fontsize=16, ha='center', va='center',
                    color=color, fontweight='bold')
            if prev_pos:
                ax.plot([prev_pos[0], x], [prev_pos[1], y], color=prev_color, lw=2)
            if (x,y) in seen_positions:
                ax.add_patch(Circle((x,y),0.2,fill=False, edgecolor=color,lw=2))
            seen_positions[(x,y)] = True
            if first_pos is None:
                first_pos = (x,y)
            last_pos = (x,y)
            prev_pos = (x,y)
            prev_color = color
    if first_pos:
        ax.add_patch(Circle(first_pos,0.2, fill=False, edgecolor='black', lw=3))
    if last_pos:
        ax.plot([last_pos[0], last_pos[0]+0.2], [last_pos[1], last_pos[1]-0.2], color='black', lw=3)
    plt.axis('off')
    plt.title("An Alchemelodic Sigil Generator", color="black")
    img = io.BytesIO()
    plt.savefig(img, format='png', facecolor=fig.get_facecolor())
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return img_base64

# ---------------- Flask Routes ----------------

@app.route("/", methods=["GET","POST"])
def planner():
    current_time = datetime.now(pytz.timezone("US/Eastern"))
    formatted_date = current_time.strftime("%b. %d, %Y")
    formatted_time = current_time.strftime("%I:%M %p")
    day_of_week = current_time.strftime("%A")
    elemental_quarter = get_elemental_quarter()
    moon_phase = get_moon_phase()
    weather = get_weather()
    planetary_hour = get_planetary_hour()

    if request.method == "POST":
        physical_condition = request.form.get("physical_condition", "")
        meditation_notes = request.form.get("meditation", "")
        tarot_notes = request.form.get("tarot", "")
        rituals_notes = {ritual: request.form.get(f"note_{ritual}", "") for ritual in RITUALS}
        completed_rituals = request.form.getlist("rituals")
        log_text = f"Date: {formatted_date}\nTime: {formatted_time}\nDay: {day_of_week}\n"
        log_text += f"Elemental Quarter: {elemental_quarter}\nMoon Phase: {moon_phase}\n"
        log_text += f"Weather: {weather}\nPlanetary Hour: {planetary_hour}\n"
        log_text += f"Physical Condition: {physical_condition}\nMeditation Notes: {meditation_notes}\n"
        log_text += f"Tarot Notes: {tarot_notes}\nRituals:\n"
        for ritual in RITUALS:
            status = "Performed" if ritual in completed_rituals else "Not Performed"
            notes = rituals_notes.get(ritual,"")
            log_text += f"  {ritual}: {status}, Notes: {notes}\n"
        log_filename = os.path.join(LOG_FOLDER, f"log_{current_time.strftime('%Y-%m-%d_%I-%M-%S_%p')}.txt")
        with open(log_filename, "w") as f:
            f.write(log_text)
        return send_file(io.BytesIO(log_text.encode("utf-8")),
                         as_attachment=True,
                         download_name=os.path.basename(log_filename),
                         mimetype="text/plain")
    return render_template("planner.html",
                           formatted_date=formatted_date,
                           formatted_time=formatted_time,
                           day_of_week=day_of_week,
                           elemental_quarter=elemental_quarter,
                           moon_phase=moon_phase,
                           weather=weather,
                           planetary_hour=planetary_hour,
                           rituals=RITUALS)

@app.route("/sigils", methods=["GET","POST"])
def sigils():
    sigil_image = None
    if request.method == "POST":
        word = request.form.get("word","").strip()
        if word:
            sigil_image = draw_rose_sigil(word)
    return render_template("sigils.html", sigil_image=sigil_image)

# ---------------- Run App ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
