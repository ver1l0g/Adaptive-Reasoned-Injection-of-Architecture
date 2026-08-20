"""Preprocess Titanic dataset for GP-NN."""
import csv, math, sys

rows = list(csv.reader(open(r'C:\Users\banny\Documents\algorithm project\raw_titanic.csv')))
header = rows[0]
data = rows[1:]

# Column indices
idx = {name: i for i, name in enumerate(header)}
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

# Extract features, encode, fill missing
ages = []
fares = []
for r in data:
    try: ages.append(float(r[idx['Age']]))
    except: pass
    try: fares.append(float(r[idx['Fare']]))
    except: pass

age_median = sorted(ages)[len(ages)//2] if ages else 28.0
fare_median = sorted(fares)[len(fares)//2] if fares else 15.0

sex_map = {'male': 0.0, 'female': 1.0}
embarked_map = {'C': 0.0, 'Q': 1.0, 'S': 2.0}

out_rows = []
for r in data:
    try:
        survived = float(r[idx['Survived']])
        pclass = float(r[idx['Pclass']])
        sex = sex_map.get(r[idx['Sex']], 0.0)
        try: age = float(r[idx['Age']])
        except: age = age_median
        sibsp = float(r[idx['SibSp']])
        parch = float(r[idx['Parch']])
        try: fare = float(r[idx['Fare']])
        except: fare = fare_median
        fare = math.log1p(fare)  # normalize skewed fare
        embarked = embarked_map.get(r[idx['Embarked']], 2.0)  # default S

        # Scale to [-1, 1] range for NEURON tanh
        pclass_n = (pclass - 2.0) / 1.0       # 1,2,3 -> -1,0,1
        age_n = (age - 28.0) / 15.0            # center at median
        sibsp_n = (sibsp - 0.5) / 1.5          # center
        parch_n = (parch - 0.4) / 1.0
        fare_n = (fare - fare_median) / 0.8    # center at log-median
        embarked_n = (embarked - 1.0) / 1.0    # 0,1,2 -> -1,0,1

        # 7 inputs: pclass, sex, age, sibsp, parch, fare, embarked
        # 1 output: survived
        out_rows.append([pclass_n, sex, age_n, sibsp_n, parch_n, fare_n, embarked_n, survived])
    except:
        pass

with open('bench_titanic.csv', 'w') as f:
    for row in out_rows:
        f.write(','.join(f'{v:.4f}' for v in row) + '\n')

print(f"Wrote bench_titanic.csv: {len(out_rows)} samples, 7 inputs, 1 output (BCE)")
print(f"Survival rate: {sum(r[7] for r in out_rows)/len(out_rows):.1%}")
