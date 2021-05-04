# Answering Neural Network (Ann)

## Instructions

Works for Python3.6
* install [requirements.txt](https://github.com/maduardar/squad/blob/master/requirements.txt)
* clone the repo
```
git clone https://github.com/maduardar/squad
cd squad/
```
* prepare data for trainig
``` 
sudo python3 prepare.py
``` 
 *Hint: If you have already downloaded `glove.840B.300d.txt`, put in in the same folder*
* For trainig: 
```
sudo python3 train.py
```
* For testing on dev-set to get F1-score:
```
sudo python3 test.py
```
* To find answer on question with given passage:
```
sudo python3 demo.py
```
