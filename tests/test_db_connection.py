from firebase_admin import credentials, firestore, initialize_app


# Test db connectivity
def test_firestore_connexion():
    try:
        cred = credentials.Certificate("serviceAccount.json")
        app = initialize_app(cred)
        db = firestore.client()

        # Test Firestore connection by fetching a collection
        docs = list(db.collection("prediction_history").limit(1).stream())
        assert isinstance(docs, list)
    except Exception as e:
        assert False, f"Failed to connect to Firestore: {e}"
