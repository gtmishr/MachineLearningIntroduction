from sklearn import tree

# First number is the weight in grams, and the second is whether they are bumpy or smooth (0 for bumpy, 1 for smooth)
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 are apples, 1 is an orange
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[150, 0]]))
print(clf.predict([[145, 1]]))
