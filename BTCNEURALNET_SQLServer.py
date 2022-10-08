{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a10744cc-a1c6-4a1f-a81a-3f885b8cc2b5",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import matplotlib.pyplot as plt \n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "%matplotlib inline\n",
    "plt.style.use('fivethirtyeight')\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from keras.layers.core import Dense, Activation, Dropout\n",
    "from keras.layers import LSTM\n",
    "from keras.models import Sequential\n",
    "import datetime\n",
    "from datetime import datetime\n",
    "from binance.client import Client "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "abda9e1f-1f73-4f39-8f1c-fb953bf66287",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import mysql.connector\n",
    "\n",
    "AccessFileMySql = r\"C:\\Users\\Vinnie\\Documents\\DataScienceProjects\\AccessFile-MySql.txt\"\n",
    "\n",
    "def GetMySqlLogin(AccessFile):\n",
    "    AF = open(AccessFile, \"r\")\n",
    "    MySqlAccess = AF.read()\n",
    "    Fields = MySqlAccess.split(\"\\n\")\n",
    "    MySql = {}\n",
    "    MySql['Name']       = Fields[0]\n",
    "    MySql['Login']      = Fields[1]\n",
    "    MySql['Password']   = Fields[2]\n",
    "    MySql['172.104.60.114']     = Fields[3]\n",
    "    MySql['Database']   = Fields[4]\n",
    "    return MySql\n",
    "\n",
    "MySql = GetMySqlLogin(AccessFileMySql)\n",
    "\n",
    "print(\"Using MySQL credentials: \", MySql['Name'])\n",
    "\n",
    "CNX = mysql.connector.connect(user=MySql['Login'], password=MySql['Password'], host=MySql['172.104.60.114'],\n",
    "                              database=MySql['Database'])\n",
    "Cursor = CNX.cursor()\n",
    "\n",
    "# Sql = f\"select OpenTime, Open, High, Low, Close, Volume from BTCUSDT_M15 order by OpenTime desc limit 500\"\n",
    "Sql =  f\"select OpenTime, Open, High, Low, Close, Volume from BTCUSDT_M1 order by OpenTime desc limit 500\"\n",
    "SqlTime = f\"select DATE_FORMAT(OpenTime, '%H:%i:%S') from BTCUSDT_M1 limit 500\"\n",
    "Cursor.execute(Sql)\n",
    "# OHLCData = Cursor.fetchone()\n",
    "OHLCData = Cursor.fetchall()\n",
    "CNX.commit()\n",
    "Cursor.execute(SqlTime)\n",
    "# OHLCData = Cursor.fetchone()\n",
    "TimeData = Cursor.fetchall()\n",
    "CNX.commit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e7db6598-0e6f-4587-bd0e-7f44e2292c8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "OHLCData = pd.DataFrame(OHLCData)\n",
    "TimeData = pd.DataFrame(TimeData)\n",
    "TimeData = TimeData.astype(str)\n",
    "OHLCData[0] = TimeData\n",
    "OHLCData[0] = OHLCData[0].astype(str)\n",
    "dt = datetime.now()\n",
    "OHLCData[0] = int(dt.strftime(\"%Y%m%d%H%M%S\"))\n",
    "OHLCData = np.array(OHLCData)\n",
    "OHLCData.reshape(3000,1)\n",
    "OHLCData.tolist()\n",
    "OHLCData"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a905f0bb-6b0c-4176-af72-9a9ce4297c5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "price = np.array([float(OHLCData[i][4]) for i in range(500)])\n",
    "time = np.array([int(OHLCData[i][0]) for i in range(500)])\n",
    "t = np.array([datetime.fromtimestamp(time[i]/1000).strftime('%H:%M:%S') for i in range(500)])\n",
    "price.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e6d396c7-e312-453b-8c83-740aff0c8f98",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8,5));\n",
    "plt.xlabel('Time Step');\n",
    "plt.ylabel('Bitcoin Price $')\n",
    "plt.plot(price);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d0c5ac47-7798-4bd4-bf88-b61469326d4d",
   "metadata": {},
   "outputs": [],
   "source": [
    "timeframe = pd.DataFrame({'Time':t,'Price $BTC':price})\n",
    "timeframe #1 minute price "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "014fc0e6-6c7c-431f-9cad-6a169dc591e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "price = price.reshape(500,1)\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(price[:374])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75d60561-d81e-46b0-a6fa-95153b45b978",
   "metadata": {},
   "outputs": [],
   "source": [
    "StandardScaler(copy=True, with_mean=True, with_std=True)\n",
    "price = scaler.transform(price)\n",
    "df = pd.DataFrame(price.reshape(100,5),columns=['First','Second','Third','Fourth','Target'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ef02e75-b679-4db5-913d-ee5fedb0b603",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = df.iloc[:74,:4]\n",
    "y_train = df.iloc[:74,-1]\n",
    "x_test = df.iloc[75:99,:4]\n",
    "y_test = df.iloc[75:99,-1]\n",
    "x_train = np.array(x_train)\n",
    "y_train = np.array(y_train)\n",
    "x_test = np.array(x_test)\n",
    "y_test = np.array(y_test)\n",
    "x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))\n",
    "x_test  = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))\n",
    "x_train.shape , x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c02feb98-abf6-4617-bc43-374848396e9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(LSTM(20, return_sequences=True, input_shape=(4, 1)))\n",
    "model.add(LSTM(40, return_sequences=False))\n",
    "model.add(Dense(1, activation='linear'))\n",
    "model.compile(loss='mse', optimizer='rmsprop')\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ad83eb5-dc7c-43be-af64-7bb768088c6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(x_train, y_train, batch_size=5,epochs=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e711637f-5354-4fa0-8731-8843ec6e555d",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(x_test)\n",
    "plt.figure(figsize=[8,5])\n",
    "plt.title('Model Fit')\n",
    "plt.xlabel('Time Step')\n",
    "plt.ylabel('Normalized Price')\n",
    "plt.plot(y_test, label='True')\n",
    "plt.plot(y_pred, label='Prediction')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10db8687-812e-4ad0-bde2-b49964798225",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = np.array(y_pred.reshape(12,2))\n",
    "y_test = np.array(y_test.reshape(12,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e80c78c-98bb-4c28-a4f9-ef0734e901fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=[8,5])\n",
    "plt.title('Model Fit')\n",
    "plt.xlabel('Time Step')\n",
    "plt.ylabel('Price')\n",
    "plt.plot(scaler.inverse_transform(y_test), label='True')\n",
    "plt.plot(scaler.inverse_transform(y_pred), label='Prediction')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6133a6fa-d0e7-43bb-a1fa-c07e41812884",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=[8,5])\n",
    "plt.title('Model Fit')\n",
    "plt.xlabel('Time Step')\n",
    "plt.ylabel('Price')\n",
    "plt.plot(scaler.inverse_transform(y_test), label='True')\n",
    "plt.plot(scaler.inverse_transform(y_pred), label='Prediction')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db413d02-9b44-4c9b-b390-fa4086fef96a",
   "metadata": {},
   "outputs": [],
   "source": [
    "testScore = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test),scaler.inverse_transform(y_pred)))\n",
    "print('Test Score: %.2f RMSE' % (testScore))\n",
    "from sklearn.metrics import r2_score\n",
    "print('RSquared :','{:.2%}'.format(r2_score(y_test,y_pred)))\n",
    "model.save(\"Bitcoin_model.h1MinBybit\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c1f3385-3af8-4fa9-a2f4-535b4e4aebdb",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import load_model\n",
    "#load model\n",
    "model = load_model('Bitcoin_model.h1MinBybit')\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1faf51cd-e596-49cd-9b50-9240297e6c37",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVR\n",
    "#75% train , 25% test\n",
    "trainX = df.iloc[:74,:4]\n",
    "trainY = df.iloc[:74,-1]\n",
    "testX = df.iloc[75:99,:4]\n",
    "testY = df.iloc[75:99,-1]\n",
    "svr_linear = SVR(kernel='linear',C=1e3, gamma=0.1)\n",
    "svr_linear.fit(trainX,trainY)\n",
    "SVR(C=1000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1,\n",
    "    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n",
    "predY = svr_linear.predict(testX)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3a696729-ba9d-4e98-932b-d57d3b024e38",
   "metadata": {},
   "outputs": [],
   "source": [
    "predY = np.array(predY.reshape(12,2))\n",
    "testY = np.array(testY.values.reshape(12,2))\n",
    "testY.shape, predY.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea6ff477-e642-4d56-8a8b-3415876158b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=[8,5])\n",
    "plt.title('Model Fit')\n",
    "plt.xlabel('Time Step')\n",
    "plt.ylabel('Price')\n",
    "plt.plot(scaler.inverse_transform(testY), label='True')\n",
    "plt.plot(scaler.inverse_transform(predY), label='Prediction')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11a486c4-6015-4d09-9fd6-230e811cb84a",
   "metadata": {},
   "outputs": [],
   "source": [
    "testScore = np.sqrt(mean_squared_error(scaler.inverse_transform(testY),scaler.inverse_transform(predY)))\n",
    "print('Test Score: %.2f RMSE' % (testScore))\n",
    "print('RSquared :','{:.2%}'.format(r2_score(testY,predY)))\n",
    "param_grid = {\"C\": [1e-2,1e-1,1e0, 1e1, 1e2, 1e3, 1e4],\n",
    "              \"gamma\": np.logspace(-2, 2, 50),\n",
    "             'epsilon':[0.1,0.2,0.5,0.3]}\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "svm_model = SVR(kernel='linear')\n",
    "grid_search = RandomizedSearchCV(svm_model,param_grid,scoring='r2',n_jobs=-1)\n",
    "grid_search.fit(trainX,trainY)\n",
    "print(grid_search.best_estimator_)\n",
    "SVR(C=10000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=100.0,\n",
    "    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n",
    "svm_model = SVR(C=10000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=100.0,\n",
    "    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)\n",
    "svm_model.fit(trainX,trainY)\n",
    "pred = svm_model.predict(testX)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42e87608-c537-4364-b0f5-f350881fa176",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred = np.array(pred.reshape(12,2))\n",
    "pred.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7cbd0c4c-177b-4ff2-8ee9-86da44cfed2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "testScore = np.sqrt(mean_squared_error(scaler.inverse_transform(testY),scaler.inverse_transform(pred)))\n",
    "print('Test Score: %.2f RMSE' % (testScore))\n",
    "print('RSquared :','{:.2%}'.format(r2_score(testY,pred)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "57c86f57-6b45-4fe6-ba0c-012c15665313",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import RidgeCV\n",
    "ridge = RidgeCV()\n",
    "ridge.fit(trainX,trainY)\n",
    "Rpred = ridge.predict(testX)\n",
    "Rpred = np.array(Rpred.reshape(12,2))\n",
    "testScore = np.sqrt(mean_squared_error(scaler.inverse_transform(testY),scaler.inverse_transform(Rpred)))\n",
    "print('Test Score :',testScore)\n",
    "print('RSquared :','{:.2%}'.format(r2_score(testY,Rpred)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "207b8a1b-6708-41e1-ae4f-5d22ed5363c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=[8,5])\n",
    "plt.title('Model Fit')\n",
    "plt.xlabel('Time Step')\n",
    "plt.ylabel('Price')\n",
    "plt.plot(scaler.inverse_transform(testY), label='True')\n",
    "plt.plot(scaler.inverse_transform(Rpred), label='Prediction')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "589cfa31-870c-4797-b5db-7b2e5b77182d",
   "metadata": {},
   "outputs": [],
   "source": [
    "normal_price = np.array([float(OHLCData[i][4]) for i in range(500)])\n",
    "data = pd.DataFrame(normal_price.reshape(100,5),columns=['First','Second','Third','Fourth','Target'])\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c327579-4900-4dd8-a49c-3069b48a24d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_r = df.iloc[:74,:4]\n",
    "y_train_r = df.iloc[:74,-1]\n",
    "x_test_r = df.iloc[75:99,:4]\n",
    "y_test_r = df.iloc[75:99,-1]\n",
    "from tpot import TPOTClassifier\n",
    "from tpot import TPOTRegressor\n",
    "tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)\n",
    "tpot.fit(x_train_r, y_train_r)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c70efef-c9c0-4343-9390-5b4baa225c30",
   "metadata": {},
   "outputs": [],
   "source": [
    "tpred = tpot.predict(x_test_r)\n",
    "testScore = np.sqrt(mean_squared_error(y_test_r,tpred))\n",
    "print('Test Score: %.2f RMSE' % (testScore))\n",
    "print('RSquared :','{:.2%}'.format(r2_score(y_test_r,tpred)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0fda93c1-af12-4fad-87b9-ed2cc929b7de",
   "metadata": {},
   "outputs": [],
   "source": [
    "tpot.export('1minBitcoinBybitSQL.py')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75d3cb78-2528-4ec4-9e62-f860067de87d",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=[8,5])\n",
    "plt.title('Model Fit')\n",
    "plt.xlabel('Time Step')\n",
    "plt.ylabel('Price')\n",
    "plt.plot(np.array(y_test_r).reshape(24,), label='True')\n",
    "plt.plot(tpred, label='Prediction')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e8124251-e411-4248-887f-64dba4d3539a",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "Sql =  f\"select OpenTime, Open, High, Low, Close, Volume from BTCUSDT_M1 order by OpenTime desc limit 500\"\n",
    "SqlTime = f\"select DATE_FORMAT(OpenTime, '%H:%i:%S') from BTCUSDT_M1 limit 500\"\n",
    "Cursor.execute(Sql)\n",
    "# OHLCData = Cursor.fetchone()\n",
    "OHLCData = Cursor.fetchall()\n",
    "CNX.commit()\n",
    "Cursor.execute(SqlTime)\n",
    "# OHLCData = Cursor.fetchone()\n",
    "TimeData = Cursor.fetchall()\n",
    "CNX.commit()\n",
    "OHLCData = pd.DataFrame(OHLCData)\n",
    "TimeData = pd.DataFrame(TimeData)\n",
    "TimeData = TimeData.astype(str)\n",
    "OHLCData[0] = TimeData\n",
    "OHLCData[0] = OHLCData[0].astype(str)\n",
    "dt = datetime.now()\n",
    "OHLCData[0] = int(dt.strftime(\"%Y%m%d%H%M%S\"))\n",
    "OHLCData"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5ac836d-ecdc-410f-a73b-0e65d6eecbee",
   "metadata": {},
   "outputs": [],
   "source": [
    "OHLCData = np.array(OHLCData)\n",
    "OHLCData.reshape(3000,1)\n",
    "OHLCData.tolist()\n",
    "index = [4,5,2,1]\n",
    "candles = scaler.transform(np.array([float(OHLCData[i][4]) for i in index]).reshape(4,-1))\n",
    "model_feed = candles.reshape(1,4,1)\n",
    "model_feed.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9f16858-ac49-4fc7-9583-85957916ba44",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(float(scaler.inverse_transform(model.predict(model_feed))[0][0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9f2d9a4b-3371-4eb4-a8e3-0d4d690fe3a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
