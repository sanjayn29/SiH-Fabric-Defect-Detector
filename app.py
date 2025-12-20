# app.py - FINAL VERSION - WORKS WITH YOUR CURRENT DATABASE
from flask import Flask, render_template, Response, redirect, request, jsonify, make_response, url_for
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
import urllib.parse
import io

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)

# ====================== DATABASE ======================
DATABASE_URL = "postgresql://postgres:N.Sanjay%402005@localhost:5432/Fab"
url = urlparse(DATABASE_URL)
db_config = {
    'dbname': url.path[1:],
    'user': url.username,
    'password': urllib.parse.unquote(url.password),
    'host': url.hostname,
    'port': url.port or 5432
}

def get_db_connection():
    return psycopg2.connect(**db_config)

# ====================== GLOBAL STATE ======================
active_batch = None  # Now stores {'batch_id': 1, 'batch_name': '...', 'fabric_type': '...'}
SAVE_DIR = "static/defects"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = 'best_model.h5'
CLASS_NAMES = ['broken_stitch', 'defect-free', 'hole', 'hole', 'lines',
               'needle_mark', 'pinched_fabric', 'stain', 'stain']
INPUT_SIZE = (224, 224)
DEFECT_THRESHOLD = 0.85

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

last_save_time = 0
MIN_SAVE_INTERVAL = 3

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize(INPUT_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_with_tta(frame):
    preds = []
    for flip in [-1, 0, 1]:
        img = frame if flip == -1 else cv2.flip(frame, flip)
        preds.append(model.predict(preprocess_image(img), verbose=0)[0])
    avg = np.mean(preds, axis=0)
    return CLASS_NAMES[np.argmax(avg)], np.max(avg)

def save_defect(defect_name, confidence, image_path):
    global last_save_time, active_batch
    if time.time() - last_save_time < MIN_SAVE_INTERVAL:
        return False
    if not active_batch:
        print("No active batch → defect not saved")
        return False

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO defects (defect_name, confidence, image_path, batch_id, detected_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (defect_name, float(confidence), image_path, active_batch['batch_id']))
        conn.commit()
        cur.close()
        conn.close()
        last_save_time = time.time()
        print(f"[SAVED] Batch #{active_batch['batch_id']} | {defect_name} ({confidence:.1%})")
        return True
    except Exception as e:
        print("DB Error:", e)
        return False

# ====================== CAMERA STREAM ======================
def generate_frames():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        placeholder = open("static/no_camera.jpg", "rb").read()
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(0.1)

    print("Camera started")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_class, confidence = predict_with_tta(frame)
        color = (0, 255, 0) if pred_class == 'defect-free' else (0, 0, 255)
        label = f"{pred_class.replace('_', ' ').title()} ({confidence:.1%})"
        cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3)

        if pred_class != 'defect-free' and confidence > DEFECT_THRESHOLD:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
            cv2.putText(frame, "DEFECT!", (frame.shape[1]//4, 160),
                        cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 0, 255), 7)

            filename = f"{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            full_path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(full_path, frame)
            web_path = f"/{SAVE_DIR}/{filename}"
            save_defect(pred_class, confidence, web_path)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# ====================== ROUTES ======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_batch', methods=['GET', 'POST'])
def start_batch():
    global active_batch
    if request.method == 'POST':
        batch_name = request.form['batch_name'].strip()
        fabric_type = request.form['fabric_type'].strip()

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO batch_details (batch_name, fabric_type, created_at, status)
                VALUES (%s, %s, NOW(), 'active')
                RETURNING batch_id, batch_name, fabric_type
            """, (batch_name, fabric_type))
            result = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()

            active_batch = {
                'batch_id': result[0],
                'batch_name': result[1],
                'fabric_type': result[2]
            }
            print(f"New batch started: #{active_batch['batch_id']} - {active_batch['batch_name']}")
            return redirect('/')

        except Exception as e:
            print("Batch creation error:", e)
            return f"<h3>Error creating batch:</h3><p>{e}</p>", 500

    return render_template('start_batch.html')

@app.route('/finish_batch')
def finish_batch():
    global active_batch
    if active_batch:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE batch_details SET status = 'completed', ended_at = NOW() WHERE batch_id = %s",
                    (active_batch['batch_id'],))
        conn.commit()
        cur.close()
        conn.close()
        print(f"Batch #{active_batch['batch_id']} finished.")
        active_batch = None
    return redirect('/')

@app.route('/current_batch')
def current_batch():
    if active_batch:
        return jsonify({
            "id": active_batch['batch_id'],
            "name": active_batch['batch_name'],
            "fabric_type": active_batch['fabric_type']
        })
    return jsonify({"id": None, "name": "No Active Batch", "fabric_type": ""})

@app.route('/defect_count')
def defect_count():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if active_batch:
            cur.execute("SELECT COUNT(*) FROM defects WHERE batch_id = %s", (active_batch['batch_id'],))
        else:
            cur.execute("SELECT COUNT(*) FROM defects")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"count": count}
    except:
        return {"count": 0}

@app.route('/defects')
def defects_list():
    batch_id = request.args.get('batch_id')
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("SELECT batch_id, batch_name, fabric_type, created_at, status FROM batch_details ORDER BY created_at DESC")
        batches = cur.fetchall()

        if batch_id:
            cur.execute("SELECT * FROM defects WHERE batch_id = %s ORDER BY detected_at DESC", (batch_id,))
        elif active_batch:
            cur.execute("SELECT * FROM defects WHERE batch_id = %s ORDER BY detected_at DESC", (active_batch['batch_id'],))
        else:
            cur.execute("SELECT * FROM defects ORDER BY detected_at DESC LIMIT 100")
        defects = cur.fetchall()

        if batch_id:
            cur.execute("SELECT defect_name, COUNT(*) as count FROM defects WHERE batch_id = %s GROUP BY defect_name", (batch_id,))
        elif active_batch:
            cur.execute("SELECT defect_name, COUNT(*) as count FROM defects WHERE batch_id = %s GROUP BY defect_name", (active_batch['batch_id'],))
        else:
            cur.execute("SELECT defect_name, COUNT(*) as count FROM defects GROUP BY defect_name")
        stats = cur.fetchall()

        labels = [row['defect_name'].replace('_', ' ').title() for row in stats]
        values = [row['count'] for row in stats]

        cur.close()
        conn.close()
    except Exception as e:
        print("Error:", e)
        defects, batches, labels, values = [], [], [], []

    return render_template('defects.html',
                          defects=defects,
                          batches=batches,
                          selected_batch=int(batch_id) if batch_id else None,
                          active_batch=active_batch,
                          defect_stats={'labels': labels, 'values': values})

@app.route('/delete_defect/<int:defect_id>')
def delete_defect(defect_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT image_path FROM defects WHERE id = %s", (defect_id,))
        row = cur.fetchone()
        if row:
            path = row[0].lstrip('/')
            if os.path.exists(path):
                os.remove(path)
            cur.execute("DELETE FROM defects WHERE id = %s", (defect_id,))
            conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(e)
    return redirect(url_for('defects_list') + (f"?batch_id={request.args.get('batch_id')}" if request.args.get('batch_id') else ""))

@app.route('/download_pdf')
def download_pdf():
    batch_id = request.args.get('batch_id')
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if batch_id:
            cur.execute("""
                SELECT d.*, b.batch_name, b.fabric_type 
                FROM defects d 
                JOIN batch_details b ON d.batch_id = b.batch_id 
                WHERE d.batch_id = %s 
                ORDER BY d.detected_at DESC
            """, (batch_id,))
        else:
            cur.execute("""
                SELECT d.*, COALESCE(b.batch_name, 'No Batch') as batch_name, 
                       COALESCE(b.fabric_type, '-') as fabric_type
                FROM defects d 
                LEFT JOIN batch_details b ON d.batch_id = b.batch_id 
                ORDER BY d.detected_at DESC
            """)
        defects = cur.fetchall()
        cur.close()
        conn.close()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.8*inch)
        elements = []
        styles = getSampleStyleSheet()

        title = f"Fabric Defect Report - Batch #{batch_id}" if batch_id else "All Defects Report"
        elements.append(Paragraph(title, styles['Title']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(f"<b>Total Defects: {len(defects)}</b>", styles['Normal']))
        elements.append(Spacer(1, 20))

        if defects:
            data = [["#", "Image", "Type", "Conf.", "Batch", "Time"]]
            for i, d in enumerate(defects, 1):
                try:
                    img_path = os.path.join(os.getcwd(), d['image_path'].lstrip('/'))
                    img = RLImage(img_path, width=1.6*inch, height=1.3*inch)
                    img = KeepInFrame(1.6*inch, 1.3*inch, [img])
                except:
                    img = Paragraph("[No Image]", styles['Normal'])
                data.append([
                    str(i),
                    img,
                    d['defect_name'].replace('_', ' ').title(),
                    f"{d['confidence']*100:.1f}%",
                    f"{d['batch_name']} ({d['fabric_type']})",
                    d['detected_at'].strftime('%d %b %H:%M')
                ])

            table = Table(data, colWidths=[0.4*inch, 1.8*inch, 1.6*inch, 0.8*inch, 2*inch, 1.4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0066cc')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 1, colors.grey),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph("No defects found.", styles['Normal']))

        doc.build(elements)
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=Defect_Report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
        return response
    except Exception as e:
        print("PDF Error:", e)
        return "Error generating PDF", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)