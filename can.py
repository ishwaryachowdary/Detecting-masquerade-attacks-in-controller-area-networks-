# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ================================
# 2. Load Dataset
# ================================
df = pd.read_csv("can_dataset.csv")

# ================================
# 3. Time Series Features
# ================================
df['mean_val'] = df[['speed','engine_temp','rpm']].mean(axis=1)
df['std_val'] = df[['speed','engine_temp','rpm']].std(axis=1)

features_cols = ['speed','engine_temp','rpm','mean_val','std_val']

# ================================
# 4. Graph Construction (IMPROVED)
# ================================
G = nx.DiGraph()

for i in range(len(df)-1):
    G.add_edge(df.loc[i,'CAN_ID'], df.loc[i+1,'CAN_ID'])

# Graph Features
df['node_count'] = len(G.nodes)
df['edge_count'] = len(G.edges)
df['avg_degree'] = np.mean([d for n, d in G.degree()])

features_cols += ['node_count','edge_count','avg_degree']

# ================================
# 🔥 NEW: GRAPH VISUALIZATION
# ================================
plt.figure(figsize=(6,5))

# show only small part (first 30 nodes)
sub_nodes = list(G.nodes)[:30]
subG = G.subgraph(sub_nodes)

nx.draw(subG, with_labels=True, node_size=500, font_size=8)
plt.title("CAN Message Graph (Sample)")

plt.show()

# ================================
# 5. Prepare Data
# ================================
X = df[features_cols]
y = df['label'].map({'Normal':0, 'Attack':1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ================================
# 6. Train Model
# ================================
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ================================
# 7. Prediction
# ================================
y_pred = model.predict(X_test)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

# ================================
# 8. Count
# ================================
normal_count = (y_test == 0).sum()
attack_count = (y_test == 1).sum()

print("\nNormal:", normal_count)
print("Attack:", attack_count)

# ================================
# 9. BAR GRAPH
# ================================
plt.figure()

labels = ['Normal', 'Attack']
counts = [normal_count, attack_count]

plt.bar(labels, counts)
plt.title("Normal vs Attack Count")
plt.xlabel("Category")
plt.ylabel("Count")

plt.show()

# ================================
# 10. Manual Prediction
# ================================
def predict_manual(speed, temp, rpm):

    if speed > 120:
        return "Attack Detected (High Speed Rule)"
    
    if speed > 100 and temp > 110:
        return "Attack Detected (Speed+Temp Rule)"

    mean_val = np.mean([speed,temp,rpm])
    std_val = np.std([speed,temp,rpm])

    features = [[speed,temp,rpm,mean_val,std_val,
                 len(G.nodes),len(G.edges),
                 np.mean([d for n, d in G.degree()])]]

    pred = model.predict(features)

    return "Normal Message" if pred[0]==0 else "Attack Detected"

# ================================
# 11. GUI
# ================================
def gui_predict():
    try:
        speed = float(entry_speed.get())
        temp = float(entry_temp.get())
        rpm = float(entry_rpm.get())

        result = predict_manual(speed,temp,rpm)
        messagebox.showinfo("Result", result)

    except:
        messagebox.showerror("Error","Invalid input")

root = tk.Tk()
root.title("CAN Attack Detection")

tk.Label(root,text="Speed (km/h)").grid(row=0,column=0)
entry_speed = tk.Entry(root)
entry_speed.grid(row=0,column=1)

tk.Label(root,text="Temperature (°C)").grid(row=1,column=0)
entry_temp = tk.Entry(root)
entry_temp.grid(row=1,column=1)

tk.Label(root,text="RPM").grid(row=2,column=0)
entry_rpm = tk.Entry(root)
entry_rpm.grid(row=2,column=1)

tk.Button(root,text="Predict",command=gui_predict).grid(row=3,columnspan=2)

root.mainloop()