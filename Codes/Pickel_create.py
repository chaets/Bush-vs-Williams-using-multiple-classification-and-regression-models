import pickle

bush = [0.15360876, 0.08041373, 0.032164883, 0.675320457]
williams = [0.117575757, 0.0, 0.0, 0.472502803]
# bush = []
# williams = []

print("PICKLING ...")
pickle.dump((bush),open('BUSH.pkl','wb'))
pickle.dump((williams),open('WILLIAMS.pkl','wb'))