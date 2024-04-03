## Application of Deep Learning in Intraday Price Prediction 
The Intraday Price Prediction system utilizes advanced deep learning techniques, specifically LSTM neural networks, to accurately forecast stock prices at 5-minute intervals. The method enhances prediction accuracy by utilizing technical indicators such as SMA and SMA10. Through meticulous data preprocessing and rigorous model training, our aim is to uncover the intricate relationships between historical stock data and indicators. Through the utilization of ensemble learning techniques and the fine-tuning of hyperparameters, the model's performance is enhanced in terms of its predictive reliability and computational efficiency. The system aims to assist investors and merchants in intraday trading decision-making by improving financial forecasting and trading procedures.

### Features
**Data Acquisition**: Collecting both historical and real-time intraday stock price data along with relevant technical indicators from reliable financial data sources and APIs.
**Feature Engineering**: Enhancing data quality and relevance through meticulous preprocessing techniques, ensuring consistency, and accuracy in the dataset.
**Model Development**: Designing and training Long Short-Term Memory (LSTM) neural network models to learn intricate temporal patterns and relationships in the preprocessed data for accurate stock price prediction.
**Evaluation and Validation**: Assessing the performance of developed models using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), etc., and validating predictions against actual stock prices to ensure reliability.
**Real-time Prediction**: Implementing mechanisms to facilitate real-time prediction of intraday stock prices, enabling timely decision-making for traders and investors.

### Requirements
- **Operating System Compatibility**: The system should be compatible with Windows, Linux, and macOS environments to ensure accessibility across different platforms.
- **Programming Language**: Development should be conducted using Python 3.6 or later versions to leverage its extensive libraries and ecosystem.
- **Libraries**: Required libraries include Pandas, NumPy, Scikit-learn, TensorFlow for deep learning, NLTK and TextBlob for natural language processing, as well as API libraries for accessing market data.
- **Integrated Development Environment (IDE)**: Used   Colab Notebook for efficient development and experimentation.

### Methodology 
- **Data Collection**: Gather intraday stock price data from reliable sources, ensuring completeness and consistency for subsequent analysis and modeling.
- **Preprocessing**: Handle missing values, outliers, and inconsistencies, and perform data normalization to prepare the dataset for further analysis.
- **EDA**: Explore the dataset to gain insights, identify patterns, and understand the relationships between variables using statistical and visual techniques.
- **Model Development**: Design an appropriate model architecture, such as LSTM, for intraday stock price prediction, considering features and target variable.
- **Training and Evaluation**: Train the model using the prepared dataset, validate its performance using suitable metrics, and fine-tune parameters for optimal results.

### System Architecture
![Screenshot 2024-04-01 234627](https://github.com/curiouzs/major_project2_nifty/assets/75234646/cd929adb-95a3-4f74-9b8e-96ea3cea4524)

### Flow Diagram
![Screenshot 2024-04-03 084346](https://github.com/curiouzs/major_project2_nifty/assets/75234646/3d353c60-2163-4163-a3ca-0941894d9d25)

### Output
![Screenshot 2024-04-02 223446](https://github.com/curiouzs/major_project2_nifty/assets/75234646/12564f16-d788-4791-9011-c17088cfc11f)

![Screenshot 2024-04-02 223415](https://github.com/curiouzs/major_project2_nifty/assets/75234646/4421c467-2f9c-4235-bf0c-351b12533be2)


### Results and Impact
- **Improved Prediction Accuracy**: The LSTM model significantly enhances intraday stock price predictions, providing traders with more reliable insights.
- **Enhanced Trading Strategies**: Accurate forecasts enable the development of more effective trading strategies, potentially increasing profitability.
- **Decision-Making**: Real-time insights streamline decision-making, allowing traders to capitalize on opportunities and mitigate risks promptly.

### Articles Published / References
[1] Shen, J., Shafiq, M.O. Short-term stock market price trend prediction using a comprehensive deep learning system. J Big Data 7, 66 (2020).

[2] Weng B, Lu L, Wang X, Megahed FM, Martinez W. Predicting short-term stock prices using ensemble methods and online data sources. Expert Syst Appl. 2018;112:258–73.

[3] Pang X, Zhou Y, Wang P, Lin W, Chang V. An innovative neural network approach for stock market prediction. J Supercomput. 2018. https://doi.org/10.1007/s11227-017-2228-y.

[4] Wang X, Lin W. Stock market prediction using neural networks: does trading volume help in short-term prediction.

[5] Mokhtari, Sohrab & Yen, Kang & Liu, Jin. (2021). Effectiveness of Artificial Intelligence in Stock Market Prediction based on Machine Learning. International Journal of Computer Applications. 183. 1-8.

[6] Mehar Vijh, Deeksha Chandola, Vinay Anand Tikkiwal, Arun Kumar, Stock Closing Price Prediction using Machine Learning Techniques, Procedia Computer Science, Volume 167, 2020, Pages 599-606, ISSN 1877-0509.

[7] Murkute, Amod, and Tanuja Sarode. "Forecasting market price of stock using artificial neural network." International Journal of Computer Applications 124.12 (2015): 11-15.

[8] Khan, Zabir & Alin, Tasnim & Hussain, Md. Akter. (2011). Price Prediction of Share Market Using Artificial Neural Network 'ANN'. International Journal of Computer Applications. 22. 42–47. 10.5120/2552-3497. 
