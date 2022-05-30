## Set Up Instructions & Description of Documents


 #### Documents & Files
 - `data folder`: Raw data from data source, originally split into Train and Test set in 75:25 ratio
 -  `.py / .pickle file`: User-defined functions/ Network Class/ stopwords list for training and test purposes. They would be imported automatically into the Jupyter Notebook (.ipnyb) for testing the model.
 - `Training_Notebook_xx.ipynb` : Codes for training MLP and SVM model respectively. All literations and training/ tuning outputs are saved for evaluation/ examination.
 - `Best_Model_testData.ipynb` : Codes for runing and testing the best performing models

#### Set up Instructions


1. Extract all the files from the zip file
2. Change directory (cd) to the extracted folder
3. Install packages from requirements.txt
4. Run `Best_Model_testData.ipynb`


#### Data Source
- Drug Review Dataset (Druglib.com) Data Set, provided by:
 `Felix Gräßer, Surya Kallumadi, Hagen Malberg, and Sebastian Zaunseder. 2018. Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning. In Proceedings of the 2018 International Conference on Digital Health (DH '18). ACM, New York, NY, USA, 121-125`
 (https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Druglib.com%29)
