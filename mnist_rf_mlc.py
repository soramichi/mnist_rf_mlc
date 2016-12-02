import numpy as np
import sklearn as sk
import functools
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# works only on Python3

def load_data():
    print("fetch MNIST dataset")
    mnist = fetch_mldata('MNIST original', data_home="./data")
    #mnist = fetch_mldata('MNIST', data_home="./data")
    mnist.data   = mnist.data.astype(np.float32)
    mnist.data  /= 255
    
    mnist.target = mnist.target.astype(np.int32)

    return mnist.data, mnist.target

def get_bit(byte, loc):
    return (byte & (1 << loc)) >> loc

# bit: 1 or 0
def set_bit(byte, loc, bit):
    if bit == 1:
        return (byte | (1 << loc))
    elif bit == 0:
        return (byte & ~(1 << loc))
    else:
        print("Error: bit argment should be 1 or 0 (%d given)" % bit)
        raise

def flip_one_bit(byte, loc):
    target_bit = get_bit(byte, loc)
    flipped_bit = (~target_bit) & 1
    ret = set_bit(byte, loc, flipped_bit)
    return ret

def get_indices(r, shape):
    ret = ()

    for i in range(0, len(shape)):
        mul = functools.reduce(lambda x,y: x*y, shape[i+1:], 1)
        ret += (int(r/mul), )
        r = int(r % mul)

    return ret        

def inject_error(np_array, error_rate):
    buff = bytearray(np_array.data.tobytes())

    total_bits = len(buff) * 8
    n_error_bits = int(total_bits * error_rate)

    for _ in range(0, n_error_bits):
        r = int(np.random.randint(0, total_bits))
        (pos, bit) = get_indices(r, (len(buff), 8))
        byte = buff[pos]
        buff[pos] = flip_one_bit(byte, bit)

    return np.ndarray(shape=np_array.shape, dtype=np_array.dtype, buffer=buff)

def confirm_result(clf, x_data, y_data):
    pred = clf.predict(x_data)
    print("confusion matrix")
    print(confusion_matrix(y_data, pred))
    print("")

    print("classification report")
    print(classification_report(y_data, pred, target_names=list(map(str, range(10)))))
    print("")

    print("accuracy")
    print(accuracy_score(y_data, pred))

def main():
    # setting
    N = 60000
    BER = 0.001
    
    x, y = load_data()

    x_train, x_test = np.split(x, [N])
    y_train, y_test = np.split(y, [N])

    print("inject bit errors. BER: %f" % BER)
    x_train = inject_error(x_train, BER)

    print("training")
    clf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1,
                                 random_state=None, verbose=0, warm_start=False, class_weight=None)

    clf = clf.fit(x_train, y_train, force_no_check=True)
    print("test")
    confirm_result(clf, x_test, y_test)

if __name__ == "__main__":
    main()
