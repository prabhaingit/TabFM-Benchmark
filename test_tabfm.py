import sys
import traceback

print("=" * 80)
print("Python Version")
print("=" * 80)
print(sys.version)

print("\n" + "=" * 80)
print("STEP 1: Importing TabFM")
print("=" * 80)

try:
    import tabfm

    print("✅ TabFM imported successfully")
    print("Module Location:", tabfm.__file__)
    print("Version:", getattr(tabfm, "__version__", "Not Available"))
except Exception as e:
    print("❌ Failed to import TabFM")
    traceback.print_exc()
    raise

print("\n" + "=" * 80)
print("STEP 2: Importing Google TabFM PyTorch Backend")
print("=" * 80)

try:
    from tabfm import TabFMClassifier
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    print("✅ Backend imported successfully")
except Exception as e:
    print("❌ Failed to import backend")
    traceback.print_exc()
    raise

print("\n" + "=" * 80)
print("STEP 3: Loading Pretrained Model")
print("=" * 80)

try:
    model = tabfm_v1_0_0.load(model_type="classification")
    print("✅ Pretrained model loaded successfully")
except Exception as e:
    print("❌ Failed to load pretrained model")
    traceback.print_exc()
    raise

print("\n" + "=" * 80)
print("STEP 4: Loading Example Dataset")
print("=" * 80)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris(as_frame=True)

X = iris.data.copy()
y = iris.target

print("Dataset Shape:", X.shape)

print("\nAdding one categorical feature...")

X["petal_length_group"] = (
    X["petal length (cm)"]
    .apply(lambda x: "Small" if x < 2 else ("Medium" if x < 5 else "Large"))
)

print(X.head())

print("\n" + "=" * 80)
print("STEP 5: Train/Test Split")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

print("\n" + "=" * 80)
print("STEP 6: Creating TabFM Classifier")
print("=" * 80)

clf = TabFMClassifier(model=model)

print("Classifier Created Successfully")

print("\n" + "=" * 80)
print("STEP 7: Fitting Model")
print("=" * 80)

clf.fit(X_train, y_train)

print("✅ Fit completed")

print("\n" + "=" * 80)
print("STEP 8: Predicting")
print("=" * 80)

predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("Prediction Shape:", predictions.shape)
print("Probability Shape:", probabilities.shape)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nFirst 10 Predictions:")
print(predictions[:10])

print("\nFirst 10 True Labels:")
print(y_test.values[:10])

print("\nFirst 5 Probability Rows:")
print(probabilities[:5])

print("\n" + "=" * 80)
print("🎉 SUCCESS! Google TabFM is working correctly.")
print("=" * 80)