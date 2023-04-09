{
  "metadata": {
    "papermill": {
      "default_parameters": {},
      "duration": 16.480765,
      "end_time": "2023-04-09T14:23:38.332319",
      "environment_variables": {},
      "exception": true,
      "input_path": "__notebook__.ipynb",
      "output_path": "__notebook__.ipynb",
      "parameters": {},
      "start_time": "2023-04-09T14:23:21.851554",
      "version": "2.4.0"
    },
    "kernelspec": {
      "name": "python",
      "display_name": "Python (Pyodide)",
      "language": "python"
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
      "version": "3.7.12"
    }
  },
  "nbformat_minor": 5,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "code",
      "source": "# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load in \n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Input data files are available in the \"../input/\" directory.\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# Any results you write to the current directory are saved as output.",
      "metadata": {
        "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
        "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:32.067678Z",
          "iopub.status.busy": "2023-04-09T14:23:32.066311Z",
          "iopub.status.idle": "2023-04-09T14:23:33.152439Z",
          "shell.execute_reply": "2023-04-09T14:23:33.151050Z"
        },
        "papermill": {
          "duration": 1.107543,
          "end_time": "2023-04-09T14:23:33.155563",
          "exception": false,
          "start_time": "2023-04-09T14:23:32.048020",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 1,
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": "/kaggle/input/titanic/train.csv\n\n/kaggle/input/titanic/test.csv\n\n/kaggle/input/titanic/gender_submission.csv\n"
        }
      ],
      "id": "c1663120"
    },
    {
      "cell_type": "code",
      "source": "train_data = pd.read_csv(\"/kaggle/input/titanic/train.csv\")\ntrain_data.head()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:33.187377Z",
          "iopub.status.busy": "2023-04-09T14:23:33.186961Z",
          "iopub.status.idle": "2023-04-09T14:23:33.234477Z",
          "shell.execute_reply": "2023-04-09T14:23:33.233414Z"
        },
        "papermill": {
          "duration": 0.066791,
          "end_time": "2023-04-09T14:23:33.237390",
          "exception": false,
          "start_time": "2023-04-09T14:23:33.170599",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 2,
      "outputs": [
        {
          "execution_count": 2,
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                           Allen, Mr. William Henry    male  35.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Cabin Embarked  \n",
              "0      0         A/5 21171   7.2500   NaN        S  \n",
              "1      0          PC 17599  71.2833   C85        C  \n",
              "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
              "3      0            113803  53.1000  C123        S  \n",
              "4      0            373450   8.0500   NaN        S  "
            ]
          },
          "metadata": {}
        }
      ],
      "id": "d63ffbc0"
    },
    {
      "cell_type": "code",
      "source": "test_data = pd.read_csv(\"/kaggle/input/titanic/test.csv\")\ntest_data.head()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:33.270772Z",
          "iopub.status.busy": "2023-04-09T14:23:33.270372Z",
          "iopub.status.idle": "2023-04-09T14:23:33.293588Z",
          "shell.execute_reply": "2023-04-09T14:23:33.292287Z"
        },
        "papermill": {
          "duration": 0.043812,
          "end_time": "2023-04-09T14:23:33.296342",
          "exception": false,
          "start_time": "2023-04-09T14:23:33.252530",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 3,
      "outputs": [
        {
          "execution_count": 3,
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>892</td>\n",
              "      <td>3</td>\n",
              "      <td>Kelly, Mr. James</td>\n",
              "      <td>male</td>\n",
              "      <td>34.5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>330911</td>\n",
              "      <td>7.8292</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>893</td>\n",
              "      <td>3</td>\n",
              "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
              "      <td>female</td>\n",
              "      <td>47.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>363272</td>\n",
              "      <td>7.0000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>894</td>\n",
              "      <td>2</td>\n",
              "      <td>Myles, Mr. Thomas Francis</td>\n",
              "      <td>male</td>\n",
              "      <td>62.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>240276</td>\n",
              "      <td>9.6875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Q</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>895</td>\n",
              "      <td>3</td>\n",
              "      <td>Wirz, Mr. Albert</td>\n",
              "      <td>male</td>\n",
              "      <td>27.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>315154</td>\n",
              "      <td>8.6625</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>896</td>\n",
              "      <td>3</td>\n",
              "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
              "      <td>female</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3101298</td>\n",
              "      <td>12.2875</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Pclass                                          Name     Sex  \\\n",
              "0          892       3                              Kelly, Mr. James    male   \n",
              "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
              "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
              "3          895       3                              Wirz, Mr. Albert    male   \n",
              "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
              "\n",
              "    Age  SibSp  Parch   Ticket     Fare Cabin Embarked  \n",
              "0  34.5      0      0   330911   7.8292   NaN        Q  \n",
              "1  47.0      1      0   363272   7.0000   NaN        S  \n",
              "2  62.0      0      0   240276   9.6875   NaN        Q  \n",
              "3  27.0      0      0   315154   8.6625   NaN        S  \n",
              "4  22.0      1      1  3101298  12.2875   NaN        S  "
            ]
          },
          "metadata": {}
        }
      ],
      "id": "0809548a"
    },
    {
      "cell_type": "code",
      "source": "women = train_data.loc[train_data.Sex == 'female'][\"Survived\"]\nrate_women = sum(women)/len(women)\n\nprint(\"% of women who survived:\", rate_women)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:33.329509Z",
          "iopub.status.busy": "2023-04-09T14:23:33.329080Z",
          "iopub.status.idle": "2023-04-09T14:23:33.341440Z",
          "shell.execute_reply": "2023-04-09T14:23:33.340097Z"
        },
        "papermill": {
          "duration": 0.032197,
          "end_time": "2023-04-09T14:23:33.344030",
          "exception": false,
          "start_time": "2023-04-09T14:23:33.311833",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 4,
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": "% of women who survived: 0.7420382165605095\n"
        }
      ],
      "id": "a65d3f02"
    },
    {
      "cell_type": "code",
      "source": "men = train_data.loc[train_data.Sex == 'male'][\"Survived\"]\nrate_men = sum(men)/len(men)\n\nprint(\"% of men who survived:\", rate_men)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:33.377136Z",
          "iopub.status.busy": "2023-04-09T14:23:33.376230Z",
          "iopub.status.idle": "2023-04-09T14:23:33.384500Z",
          "shell.execute_reply": "2023-04-09T14:23:33.383251Z"
        },
        "papermill": {
          "duration": 0.028058,
          "end_time": "2023-04-09T14:23:33.387464",
          "exception": false,
          "start_time": "2023-04-09T14:23:33.359406",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 5,
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": "% of men who survived: 0.18890814558058924\n"
        }
      ],
      "id": "5dacee49"
    },
    {
      "cell_type": "code",
      "source": "from sklearn.ensemble import RandomForestClassifier\n\ny = train_data[\"Survived\"]\n\nfeatures = [\"Pclass\", \"Sex\", \"SibSp\", \"Parch\"]\nX = pd.get_dummies(train_data[features])\nX_test = pd.get_dummies(test_data[features])\n\nmodel = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)\nmodel.fit(X, y)\npredictions = model.predict(X_test)\n\noutput = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})\noutput.to_csv('submission.csv', index=False)\nprint(\"Your submission was successfully saved!\")",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:33.420916Z",
          "iopub.status.busy": "2023-04-09T14:23:33.419703Z",
          "iopub.status.idle": "2023-04-09T14:23:34.175923Z",
          "shell.execute_reply": "2023-04-09T14:23:34.173540Z"
        },
        "papermill": {
          "duration": 0.775688,
          "end_time": "2023-04-09T14:23:34.178520",
          "exception": false,
          "start_time": "2023-04-09T14:23:33.402832",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 6,
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": "Your submission was successfully saved!\n"
        }
      ],
      "id": "8b96af72"
    },
    {
      "cell_type": "code",
      "source": "train = pd.read_csv(\"/kaggle/input/titanic/train.csv\")\ntest = pd.read_csv(\"/kaggle/input/titanic/test.csv\")",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:34.211936Z",
          "iopub.status.busy": "2023-04-09T14:23:34.211458Z",
          "iopub.status.idle": "2023-04-09T14:23:34.231263Z",
          "shell.execute_reply": "2023-04-09T14:23:34.230066Z"
        },
        "papermill": {
          "duration": 0.040074,
          "end_time": "2023-04-09T14:23:34.234185",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.194111",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 7,
      "outputs": [],
      "id": "74ee1d50"
    },
    {
      "cell_type": "markdown",
      "source": "데이터는 이미 훈련용과 테스트용이 나눠져 있음. 학습을 위해 따로 분류할 필요는 없음.\n\n데이터는 타이타닉호 탑승객들의 신상정보를 series로 갖는 dataframe으로 구성되어있다. 승객별로 생존여부가 모두 라벨링되어 있으므로 타이타닉 문제는 지도학습에 해당한다. 최종적으로 생존(1)과 사망(0)으로 분류하였다.",
      "metadata": {
        "papermill": {
          "duration": 0.015251,
          "end_time": "2023-04-09T14:23:34.264931",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.249680",
          "status": "completed"
        },
        "tags": []
      },
      "id": "9dfd11db"
    },
    {
      "cell_type": "code",
      "source": "train.shape",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:34.298033Z",
          "iopub.status.busy": "2023-04-09T14:23:34.297579Z",
          "iopub.status.idle": "2023-04-09T14:23:34.305411Z",
          "shell.execute_reply": "2023-04-09T14:23:34.304096Z"
        },
        "papermill": {
          "duration": 0.027602,
          "end_time": "2023-04-09T14:23:34.308059",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.280457",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 8,
      "outputs": [
        {
          "execution_count": 8,
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(891, 12)"
            ]
          },
          "metadata": {}
        }
      ],
      "id": "c6a81373"
    },
    {
      "cell_type": "code",
      "source": "test.shape",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:34.341913Z",
          "iopub.status.busy": "2023-04-09T14:23:34.341042Z",
          "iopub.status.idle": "2023-04-09T14:23:34.348050Z",
          "shell.execute_reply": "2023-04-09T14:23:34.346844Z"
        },
        "papermill": {
          "duration": 0.02672,
          "end_time": "2023-04-09T14:23:34.350608",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.323888",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 9,
      "outputs": [
        {
          "execution_count": 9,
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(418, 11)"
            ]
          },
          "metadata": {}
        }
      ],
      "id": "7e80d296"
    },
    {
      "cell_type": "markdown",
      "source": "훈련데이터는 총 891개의 행과 12개의 열로 되어있다.\n테스트는 418개의 행과 11개의 열로 되어있는데 이는 Survived의 열이 빠진 상태이기 때문이다.",
      "metadata": {
        "papermill": {
          "duration": 0.016815,
          "end_time": "2023-04-09T14:23:34.383279",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.366464",
          "status": "completed"
        },
        "tags": []
      },
      "id": "ad6f4650"
    },
    {
      "cell_type": "code",
      "source": "train.info()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:34.416855Z",
          "iopub.status.busy": "2023-04-09T14:23:34.416409Z",
          "iopub.status.idle": "2023-04-09T14:23:34.434064Z",
          "shell.execute_reply": "2023-04-09T14:23:34.432523Z"
        },
        "papermill": {
          "duration": 0.037843,
          "end_time": "2023-04-09T14:23:34.436779",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.398936",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 10,
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": "<class 'pandas.core.frame.DataFrame'>\n\nRangeIndex: 891 entries, 0 to 890\n\nData columns (total 12 columns):\n\n #   Column       Non-Null Count  Dtype  \n\n---  ------       --------------  -----  \n\n 0   PassengerId  891 non-null    int64  \n\n 1   Survived     891 non-null    int64  \n\n 2   Pclass       891 non-null    int64  \n\n 3   Name         891 non-null    object \n\n 4   Sex          891 non-null    object \n\n 5   Age          714 non-null    float64\n\n 6   SibSp        891 non-null    int64  \n\n 7   Parch        891 non-null    int64  \n\n 8   Ticket       891 non-null    object \n\n 9   Fare         891 non-null    float64\n\n 10  Cabin        204 non-null    object \n\n 11  Embarked     889 non-null    object \n\ndtypes: float64(2), int64(5), object(5)\n\nmemory usage: 83.7+ KB\n"
        }
      ],
      "id": "3ea6e342"
    },
    {
      "cell_type": "code",
      "source": "test.info()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:34.471701Z",
          "iopub.status.busy": "2023-04-09T14:23:34.470216Z",
          "iopub.status.idle": "2023-04-09T14:23:34.485324Z",
          "shell.execute_reply": "2023-04-09T14:23:34.484111Z"
        },
        "papermill": {
          "duration": 0.035145,
          "end_time": "2023-04-09T14:23:34.487863",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.452718",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 11,
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": "<class 'pandas.core.frame.DataFrame'>\n\nRangeIndex: 418 entries, 0 to 417\n\nData columns (total 11 columns):\n\n #   Column       Non-Null Count  Dtype  \n\n---  ------       --------------  -----  \n\n 0   PassengerId  418 non-null    int64  \n\n 1   Pclass       418 non-null    int64  \n\n 2   Name         418 non-null    object \n\n 3   Sex          418 non-null    object \n\n 4   Age          332 non-null    float64\n\n 5   SibSp        418 non-null    int64  \n\n 6   Parch        418 non-null    int64  \n\n 7   Ticket       418 non-null    object \n\n 8   Fare         417 non-null    float64\n\n 9   Cabin        91 non-null     object \n\n 10  Embarked     418 non-null    object \n\ndtypes: float64(2), int64(4), object(5)\n\nmemory usage: 36.0+ KB\n"
        }
      ],
      "id": "0da14713"
    },
    {
      "cell_type": "markdown",
      "source": "info()를 통해 데이터프레임에 대한 더 자세한 정보를 얻을 수 있다.",
      "metadata": {
        "papermill": {
          "duration": 0.015618,
          "end_time": "2023-04-09T14:23:34.519533",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.503915",
          "status": "completed"
        },
        "tags": []
      },
      "id": "f1fc7485"
    },
    {
      "cell_type": "code",
      "source": "def chart(feature):\n    survived = train[train['Survived']==1][feature].value_counts()\n    dead = train[train['Survived']==0][feature].value_counts()\n    df = pd.DataFrame([survived, dead])\n    df.index = ['Survived', 'Dead']\n    df.plot(kind='bar', stacked=True, figsize=(10,5))",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:34.553567Z",
          "iopub.status.busy": "2023-04-09T14:23:34.553111Z",
          "iopub.status.idle": "2023-04-09T14:23:34.560041Z",
          "shell.execute_reply": "2023-04-09T14:23:34.558989Z"
        },
        "papermill": {
          "duration": 0.02706,
          "end_time": "2023-04-09T14:23:34.562528",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.535468",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 12,
      "outputs": [],
      "id": "d0049b99"
    },
    {
      "cell_type": "markdown",
      "source": "각 항목들을 시각화하여 대략적인 상태를 파악하고 계수들끼리의 상관관계를 알아보기 위해 막대 그래프로 시각화하였다. 위의 함수는 막대 그래프로 그려주는 함수다.",
      "metadata": {
        "papermill": {
          "duration": 0.015635,
          "end_time": "2023-04-09T14:23:34.594098",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.578463",
          "status": "completed"
        },
        "tags": []
      },
      "id": "14a02394"
    },
    {
      "cell_type": "code",
      "source": "chart('Sex')\nchart('Pclass')\nchart('Embarked')\nchart('SibSp')",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:34.628095Z",
          "iopub.status.busy": "2023-04-09T14:23:34.627207Z",
          "iopub.status.idle": "2023-04-09T14:23:35.617624Z",
          "shell.execute_reply": "2023-04-09T14:23:35.616630Z"
        },
        "papermill": {
          "duration": 1.01055,
          "end_time": "2023-04-09T14:23:35.620567",
          "exception": false,
          "start_time": "2023-04-09T14:23:34.610017",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAz8AAAHbCAYAAADlHyT+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAuIklEQVR4nO3dfZjVdZ3/8ddwN9zNjHI346xog6GbC5pil8JWUAJq3mTW4qa1WrRraWyIXCrZbmReoO7mTetmdxZqS9S22dalIZSGsa4bYqaia4UouM5IJs6AwowO5/dHl+fXiJKjyJnh+3hc17muOd/v55x5H8jO9eR7zvdbVSqVSgEAANjD9an0AAAAALuD+AEAAApB/AAAAIUgfgAAgEIQPwAAQCGIHwAAoBDEDwAAUAj9Kj3Aa7F9+/Y88cQTqampSVVVVaXHAQAAKqRUKmXz5s1pbGxMnz47P7bTK+PniSeeyOjRoys9BgAA0ENs2LAh++67707X9Mr4qampSfKHF1hbW1vhaQAAgEppa2vL6NGjy42wM70yfl78qFttba34AQAAXtXXYZzwAAAAKATxAwAAFIL4AQAACqFXfufn1ers7Mzzzz9f6TEKacCAAX/yVIMAALA77ZHxUyqV0tLSkmeeeabSoxRWnz590tTUlAEDBlR6FAAASLKHxs+L4TNq1KgMHjzYhVB3sxcvQtvc3Jz99tvPnz8AAD3CHhc/nZ2d5fAZPnx4pccprJEjR+aJJ57ICy+8kP79+1d6HAAA2PNOePDid3wGDx5c4UmK7cWPu3V2dlZ4EgAA+IM9Ln5e5KNWleXPHwCAnmaPjR8AAIA/Jn4AAIBC2ONOeLAzb7rw5t36+x699PhurS+VSjnrrLPyve99L5s2bcovf/nLvPWtb31jhtuJRx99NE1NTRX7/QAA8EYoVPz0dEuXLs2iRYvys5/9LGPGjMmIESMqPRIAAOwxxE8Psnbt2uyzzz6ZNGlSpUcBAIA9ju/89BBnnnlmZs2alfXr16eqqipvetObUiqVcvnll2fMmDEZNGhQDj300Hzve98rP+ZnP/tZqqqqcuutt+awww7LoEGD8u53vzsbN27Mj3/847zlLW9JbW1tPvjBD+a5554rP27p0qV5+9vfnr322ivDhw/PCSeckLVr1+50vgcffDDvec97MnTo0NTX1+fDH/5wnnrqqTfszwMAAHY18dNDXH311bn44ouz7777prm5OatWrcpnPvOZfPOb38y1116bNWvW5Nxzz82HPvShrFixostj58+fn2uuuSZ33nlnNmzYkBkzZuSqq67K4sWLc/PNN2f58uX5l3/5l/L6Z599NnPmzMmqVavy05/+NH369Mn73ve+bN++/WVna25uzuTJk/PWt741d999d5YuXZonn3wyM2bMeEP/TAAAYFfysbceoq6uLjU1Nenbt28aGhry7LPP5oorrshtt92WiRMnJknGjBmTlStX5itf+UomT55cfuwll1ySv/zLv0ySzJw5M/PmzcvatWszZsyYJMkHPvCB3H777bnggguSJO9///u7/O7rrrsuo0aNyoMPPphx48btMNu1116bww8/PAsWLChv+8Y3vpHRo0fn17/+dQ488MBd+4cBQO8yv67SE0BlzW+t9AS8SuKnh3rwwQezbdu2TJs2rcv2jo6OHHbYYV22HXLIIeWf6+vrM3jw4HL4vLjtF7/4Rfn+2rVr8w//8A+566678tRTT5WP+Kxfv/5l42f16tW5/fbbM3To0B32rV27VvwAANAriJ8e6sUgufnmm/Nnf/ZnXfZVV1d3ud+/f//yz1VVVV3uv7jtjz/SduKJJ2b06NH52te+lsbGxmzfvj3jxo1LR0fHK85y4okn5rLLLtth3z777NO9FwYAABUifnqogw8+ONXV1Vm/fn2Xj7i9Xr///e/z0EMP5Stf+Ure8Y53JElWrly508ccfvjh+Y//+I+86U1vSr9+/icDAEDv5IQHPVRNTU3mzp2bc889N9dff33Wrl2bX/7yl/nXf/3XXH/99a/5effee+8MHz48X/3qV/Pb3/42t912W+bMmbPTx5xzzjl5+umn88EPfjC/+MUv8sgjj2TZsmX56Ec/ms7Oztc8CwAA7E6F+mf8Ry89vtIjdMvnP//5jBo1KgsXLswjjzySvfbaK4cffng+/elPv+bn7NOnT5YsWZK///u/z7hx43LQQQfli1/8YqZMmfKKj2lsbMx//dd/5YILLsgxxxyT9vb27L///jn22GPTp49+BgCgd6gqlUqlSg/RXW1tbamrq0tra2tqa2u77Nu2bVvWrVuXpqamDBw4sEIT4u8BoECc7Y2ic7a3itpZG7yUf7YHAAAKQfwAAACFIH4AAIBCED8AAEAhiB8AAKAQxA8A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73PxxRenubk548aNyy233JL999+/UiNBr1JdXZ3PfvazO3wkFIBi8X4Ar15V6dWcEw4AAKCXq9hFTgEAAHYn8QMAABSC+AEAAApB/AAAAIUgfgAAgEIQPwAAQCGIHwAAoBAqdpFT4NU55ZRTXvXa73//+2/gJABU0he/+MVXvfbv//7v38BJoPcSP9DD1dXVlX8ulUq56aabUldXlyOOOCJJsnr16jzzzDPdiiQAep8rr7yyy/3f/e53ee6557LXXnslSZ555pkMHjw4o0aNEj/wCsQP9HDf/OY3yz9fcMEFmTFjRr785S+nb9++SZLOzs6cffbZqa2trdSIAOwG69atK/+8ePHifOlLX8p1112Xgw46KEny8MMP52//9m9z1llnVWpE6PGqSqVSqdJDAK/OyJEjs3LlyvIb3YsefvjhTJo0Kb///e8rNBkAu9MBBxyQ733veznssMO6bF+9enU+8IEPdAkl4P9zwgPoRV544YU89NBDO2x/6KGHsn379gpMBEAlNDc35/nnn99he2dnZ5588skKTAS9g4+9QS/ykY98JB/96Efz29/+NkcddVSS5K677sqll16aj3zkIxWeDoDd5eijj87f/u3f5rrrrsuECRNSVVWVu+++O2eddVamTp1a6fGgx/KxN+hFtm/fnn/+53/O1Vdfnebm5iTJPvvsk0996lM577zzyt8DAmDP9rvf/S5nnHFGli5dmv79+yf5w6cDjjnmmCxatCijRo2q8ITQM4kf6KXa2tqSxIkOAArs17/+df73f/83pVIpb3nLW3LggQdWeiTo0cQP9DIvvPBCfvazn2Xt2rU57bTTUlNTkyeeeCK1tbUZOnRopccDAOixxA/0Io899liOPfbYrF+/Pu3t7fn1r3+dMWPGZPbs2dm2bVu+/OUvV3pEAHaTxx9/PD/84Q+zfv36dHR0dNl3xRVXVGgq6Nmc8AB6kU996lM54ogj8qtf/SrDhw8vb3/f+96Xj33sYxWcDIDd6ac//WlOOumkNDU15eGHH864cePy6KOPplQq5fDDD6/0eNBjOdU19CIrV67MZz7zmQwYMKDL9v333z//93//V6GpANjd5s2bl/POOy8PPPBABg4cmP/4j//Ihg0bMnny5PzVX/1VpceDHkv8QC+yffv2dHZ27rD98ccfT01NTQUmAqASHnrooZxxxhlJkn79+mXr1q0ZOnRoLr744lx22WUVng56LvEDvci0adNy1VVXle9XVVVly5Yt+exnP5v3vOc9lRsMgN1qyJAhaW9vT5I0NjZm7dq15X1PPfVUpcaCHs93fqAXufLKK/Oud70rBx98cLZt25bTTjstv/nNbzJixIh8+9vfrvR4AOwmRx11VP7rv/4rBx98cI4//vicd955uf/++/P973+/fBFsYEfO9ga9zNatW/Ptb38799xzT7Zv357DDz88p59+egYNGlTp0QDYTR555JFs2bIlhxxySJ577rnMnTs3K1euzJvf/OZceeWV2X///Ss9IvRI4gd6keeeey6DBw+u9BgAAL2S7/xALzJq1Kh86EMfyq233prt27dXehwAKuiZZ57J17/+9cybNy9PP/10kuSee+5x9k/YCfEDvcgNN9yQ9vb2vO9970tjY2M+9alPZdWqVZUeC4Dd7L777suBBx6Yyy67LP/8z/+cZ555Jkly0003Zd68eZUdDnow8QO9yCmnnJJ///d/z5NPPpmFCxfmoYceyqRJk3LggQfm4osvrvR4AOwmc+bMyZlnnpnf/OY3GThwYHn7cccdlzvuuKOCk0HP5js/0Ms9+OCDOf3003Pfffe97DWAANjz1NXV5Z577skBBxyQmpqa/OpXv8qYMWPy2GOP5aCDDsq2bdsqPSL0SI78QC+0bdu2fPe7383JJ5+cww8/PL///e8zd+7cSo8FwG4ycODAtLW17bD94YcfzsiRIyswEfQO4gd6kWXLluWMM85IfX19Pv7xj2fUqFG59dZbs379elf0BiiQ9773vbn44ovz/PPPJ/nDRa/Xr1+fCy+8MO9///srPB30XD72Br3I4MGDc/zxx+f000/P8ccfn/79+1d6JAAqoK2tLe95z3uyZs2abN68OY2NjWlpacnEiRNzyy23ZMiQIZUeEXok8QO9SFtbW2prays9BgA9xO23357Vq1eXL3o9derUSo8EPVq/Sg8A7NxLg+flPuP9ImEEsOfbvn17Fi1alO9///t59NFHU1VVlaampjQ0NKRUKqWqqqrSI0KP5cgP9HB9+/ZNc3NzRo0alT59+rzsm9qLb3bO9gawZyuVSjnxxBNzyy235NBDD82f//mfp1Qq5aGHHsr999+fk046KT/4wQ8qPSb0WI78QA932223ZdiwYeWf/YseQHEtWrQod9xxR37605/mXe96V5d9t912W04++eTccMMN+Zu/+ZsKTQg9myM/AAC9xPTp0/Pud787F1544cvuX7BgQVasWJFbb711N08GvYNTXUMvMmbMmPzDP/xDHn744UqPAkAF3HfffTn22GNfcf9xxx2XX/3qV7txIuhdxA/0Ip/85CezdOnSvOUtb8mECRNy1VVXpbm5udJjAbCbPP3006mvr3/F/fX19dm0adNunAh6F/EDvcicOXOyatWq/O///m9OOOGEXHvttdlvv/0yffr03HDDDZUeD4A3WGdnZ/r1e+WvbPft2zcvvPDCbpwIehff+YFe7q677sonPvGJ3Hfffc72BrCH69OnT4477rhUV1e/7P729vYsXbrU+wG8Amd7g17qF7/4RRYvXpzvfOc7aW1tzQc+8IFKjwTAG+yMM874k2uc6Q1emSM/0Iv8+te/zr/9279l8eLFefTRR/Oud70rp59+ek455ZTU1NRUejwAgB5N/EAv0qdPnxxxxBE57bTT8td//ddpaGio9EgAAL2G+IFeorOzM9ddd10+8IEPlC96CgDAqyd+oBcZOHBgHnrooTQ1NVV6FACAXseprqEXGT9+fB555JFKjwEA0Cs58gO9yLJly3LBBRfk85//fCZMmJAhQ4Z02V9bW1uhyQAAej7xA71Inz7//2BtVVVV+edSqZSqqirXdQAA2AnX+YFe5Pbbb6/0CAAAvZYjPwAAQCE48gO9yB133LHT/e985zt30yQAAL2PIz/Qi/zxd35e9Mff/fGdHwCAV+ZU19CLbNq0qctt48aNWbp0ad72trdl2bJllR4PAKBHc+QH9gB33HFHzj333KxevbrSowAA9FiO/MAeYOTIkXn44YcrPQYAQI/mhAfQi9x3331d7pdKpTQ3N+fSSy/NoYceWqGpAAB6Bx97g16kT58+qaqqykv/sz3qqKPyjW98I3/+539eockAAHo+8QO9yGOPPdblfp8+fTJy5MgMHDiwQhMBAPQevvMDvcD//M//5Mc//nH233//8m3FihV55zvfmf322y9/93d/l/b29kqPCQDQo4kf6AXmz5/f5fs+999/f2bOnJmpU6fmwgsvzI9+9KMsXLiwghMCAPR8PvYGvcA+++yTH/3oRzniiCOSJBdddFFWrFiRlStXJkn+/d//PZ/97Gfz4IMPVnJMAIAezZEf6AU2bdqU+vr68v0VK1bk2GOPLd9/29velg0bNlRiNACAXkP8QC9QX1+fdevWJUk6Ojpyzz33ZOLEieX9mzdvTv/+/Ss1HgBAryB+oBc49thjc+GFF+bnP/955s2bl8GDB+cd73hHef99992XAw44oIITAgD0fC5yCr3AJZdcklNOOSWTJ0/O0KFDc/3112fAgAHl/d/4xjcyffr0Ck4IANDzOeEB9CKtra0ZOnRo+vbt22X7008/naFDh3YJIgAAuhI/AABAIfjODwAAUAjiBwAAKATxAwAAFIL4AQAACkH8AAAAhSB+AACAQhA/AABAIfw/AvaK8+fXAgkAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 1000x500 with 1 Axes>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAz8AAAHbCAYAAADlHyT+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAp2klEQVR4nO3dfZTWdZ3/8dfAwHAjMwnCjHMcXTSyDPQUdBROKSag5E3lttbitlTUWiorCxyV3N3I7YBaK9bxZFtr4c0SbbvSzfkZQjdirMddxAy8yaxQYZ0RU5wBxRmF6/dHx2sbEXME5prL7+NxznUO1/f6zHzfV+fUnGef6/p+a0qlUikAAABvcP0qPQAAAEBvED8AAEAhiB8AAKAQxA8AAFAI4gcAACgE8QMAABSC+AEAAAqhttIDvB67d+/O448/nmHDhqWmpqbS4wAAABVSKpWyffv2NDc3p1+/V9/bqcr4efzxx9PS0lLpMQAAgD5i8+bNOeyww151TVXGz7Bhw5L84Q3W19dXeBoAAKBSOjo60tLSUm6EV1OV8fPSR93q6+vFDwAA8Jq+DuOCBwAAQCGIHwAAoBDEDwAAUAhV+Z0fAADg/+zevTtdXV2VHuOAGThw4J+8jPVrIX4AAKCKdXV1ZdOmTdm9e3elRzlg+vXrl9GjR2fgwIH79HvEDwAAVKlSqZTW1tb0798/LS0t+2V3pK/ZvXt3Hn/88bS2tubwww9/TVd12xvxAwAAVerFF1/Mc889l+bm5gwZMqTS4xwwI0eOzOOPP54XX3wxAwYMeN2/542XhgAAUBC7du1Kkn3+OFhf99L7e+n9vl7iBwAAqty+fBSsGuyv9yd+AACAQhA/AABAIbjgAQAAvMH82aX/r1fP98gVp/fq+V4vOz8AAECvu+OOO3LmmWemubk5NTU1+d73vnfAzyl+AACAXvfss8/muOOOy7XXXttr5/SxNwAAoNdNnz4906dP79Vz2vkBAAAKwc4PALBPxt0wrtIjQEVtnLmx0iPwGtn5AQAACkH8AAAAhSB+AACAQvCdHwAAoNft2LEjv/nNb8rPN23alHvvvTfDhw/P4YcffkDOKX4AAOAN5pErTq/0CH/S3XffnZNPPrn8fO7cuUmSmTNnZunSpQfknOIHAADodZMnT06pVOrVc/rODwAAUAjiBwAAKATxAwAAFIL4AQAACkH8AAAAhSB+AACAQhA/AABAIYgfAACgEMQPAABQCLWVHgAAANjPFjb08vnae7R88eLFueWWW/KrX/0qgwcPzqRJk3LllVfm6KOPPkAD/oGdHwAAoFetWbMmF1xwQe66666sXr06L774YqZNm5Znn332gJ7Xzg8AANCrVq5c2e35t771rYwaNSrr16/PiSeeeMDOa+cHAACoqPb2P3xsbvjw4Qf0POIHAAComFKplLlz5+bd7353xo4de0DP5WNvAABAxVx44YXZsGFD1q5de8DPJX4AAICKmD17dn7wgx/kjjvuyGGHHXbAzyd+AACAXlUqlTJ79uysWLEit99+e0aPHt0r5xU/AABAr7rggguybNmyfP/738+wYcPS1taWJGloaMjgwYMP2Hld8AAAAOhV1113Xdrb2zN58uQceuih5cd3vvOdA3peOz8AAPBGs7C90hO8qlKpVJHz2vkBAAAKQfwAAACF0KP4WbhwYWpqaro9mpqayq+XSqUsXLgwzc3NGTx4cCZPnpz777+/2+/o7OzM7Nmzc8ghh2To0KE566yzsmXLlv3zbgAAAPaixzs/b3/729Pa2lp+bNy4sfzaVVddlauvvjrXXntt1q1bl6ampkydOjXbt28vr5kzZ05WrFiR5cuXZ+3atdmxY0fOOOOM7Nq1a/+8IwAAgFfQ4wse1NbWdtvteUmpVMo111yTyy67LGeffXaS5IYbbkhjY2OWLVuW8847L+3t7bn++utz0003ZcqUKUmSm2++OS0tLfnxj3+cU0899RXP2dnZmc7OzvLzjo6Ono4NAAAUXI93fh5++OE0Nzdn9OjR+chHPpLf/e53SZJNmzalra0t06ZNK6+tq6vLSSedlDvvvDNJsn79+rzwwgvd1jQ3N2fs2LHlNa9k8eLFaWhoKD9aWlp6OjYAAFBwPYqf448/PjfeeGNuu+22fOMb30hbW1smTZqUp556qnxjosbGxm4/09jYWH6tra0tAwcOzMEHH7zXNa9kwYIFaW9vLz82b97ck7EBAAB69rG36dOnl/89bty4TJw4MUcddVRuuOGGnHDCCUmSmpqabj9TKpX2OPZyf2pNXV1d6urqejIqAABAN/t0qeuhQ4dm3Lhxefjhh8vfA3r5Ds7WrVvLu0FNTU3p6urKtm3b9roGAADgQNin+Ons7MyDDz6YQw89NKNHj05TU1NW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vOlNb0qSPPPMMxkyZEhGjRolfmAvxA/0cd/61rfK/77kkktyzjnn5Gtf+1r69++fJNm1a1fOP//81NfXV2pEAHrBpk2byv9etmxZvvrVr+b666/P0UcfnSR56KGH8qlPfSrnnXdepUaEPq+mVCqVKj0E8NqMHDkya9euLf+he8lDDz2USZMm5amnnqrQZAD0pqOOOir/8R//kXe84x3djq9fvz4f+tCHuoUS8H9c8ACqyIsvvpgHH3xwj+MPPvhgdu/eXYGJAKiE1tbWvPDCC3sc37VrV5544okKTATVwcfeoIp8/OMfzyc+8Yn85je/yQknnJAkueuuu3LFFVfk4x//eIWnA6C3nHLKKfnUpz6V66+/PuPHj09NTU3uvvvunHfeeZkyZUqlx4M+y8feoIrs3r07X/rSl/LlL385ra2tSZJDDz00F110UebNm1f+HhAAb2xPPvlkZs6cmZUrV2bAgAFJ/vDpgFNPPTVLly7NqFGjKjwh9E3iB6pUR0dHkrjQAUCB/frXv86vfvWrlEqlvO1tb8tb3vKWSo8EfZr4gSrz4osv5vbbb89vf/vbzJgxI8OGDcvjjz+e+vr6HHTQQZUeDwCgzxI/UEUeffTRnHbaaXnsscfS2dmZX//61znyyCMzZ86cPP/88/na175W6REB6CVbtmzJD37wgzz22GPp6urq9trVV19doamgb3PBA6giF110USZMmJBf/vKXGTFiRPn4Bz/4wXzyk5+s4GQA9Kaf/OQnOeusszJ69Og89NBDGTt2bB555JGUSqW8853vrPR40Ge51DVUkbVr1+bv//7vM3DgwG7HjzjiiPzv//5vhaYCoLctWLAg8+bNy3333ZdBgwblP//zP7N58+acdNJJ+Yu/+ItKjwd9lviBKrJ79+7s2rVrj+NbtmzJsGHDKjARAJXw4IMPZubMmUmS2tra7Ny5MwcddFAuv/zyXHnllRWeDvou8QNVZOrUqbnmmmvKz2tqarJjx4587nOfy/ve977KDQZArxo6dGg6OzuTJM3Nzfntb39bfu33v/99pcaCPs93fqCKLFmyJCeffHKOOeaYPP/885kxY0YefvjhHHLIIfn2t79d6fEA6CUnnHBC/uu//ivHHHNMTj/99MybNy8bN27MLbfcUr4JNrAnV3uDKrNz5858+9vfzj333JPdu3fnne98Z84999wMHjy40qMB0Et+97vfZceOHTn22GPz3HPPZf78+Vm7dm3e/OY3Z8mSJTniiCMqPSL0SeIHqshzzz2XIUOGVHoMAICq5Ds/UEVGjRqVv/qrv8ptt92W3bt3V3ocACromWeeyb/+679mwYIFefrpp5Mk99xzj6t/wqsQP1BFbrzxxnR2duaDH/xgmpubc9FFF2XdunWVHguAXrZhw4a85S1vyZVXXpkvfelLeeaZZ5IkK1asyIIFCyo7HPRh4geqyNlnn53vfve7eeKJJ7J48eI8+OCDmTRpUt7ylrfk8ssvr/R4APSSuXPn5mMf+1gefvjhDBo0qHx8+vTpueOOOyo4GfRtvvMDVe6BBx7Iueeemw0bNrziPYAAeONpaGjIPffck6OOOirDhg3LL3/5yxx55JF59NFHc/TRR+f555+v9IjQJ9n5gSr0/PPP59///d/zgQ98IO985zvz1FNPZf78+ZUeC4BeMmjQoHR0dOxx/KGHHsrIkSMrMBFUB/EDVWTVqlWZOXNmGhsb8+lPfzqjRo3Kbbfdlscee8wdvQEK5P3vf38uv/zyvPDCC0n+cNPrxx57LJdeemn+/M//vMLTQd/lY29QRYYMGZLTTz895557bk4//fQMGDCg0iMBUAEdHR153/vel/vvvz/bt29Pc3Nz2traMnHixNx6660ZOnRopUeEPkn8QBXp6OhIfX19pccAoI/42c9+lvXr15dvej1lypRKjwR9Wm2lBwBe3cuD55U+4/0SYQTwxrd79+4sXbo0t9xySx555JHU1NRk9OjRaWpqSqlUSk1NTaVHhD7Lzg/0cf37909ra2tGjRqVfv36veIftZf+2LnaG8AbW6lUyplnnplbb701xx13XN761remVCrlwQcfzMaNG3PWWWfle9/7XqXHhD7Lzg/0cT/96U8zfPjw8r/9P3oAxbV06dLccccd+clPfpKTTz6522s//elP84EPfCA33nhj/vqv/7pCE0LfZucHAKBKTJs2Le9973tz6aWXvuLrixYtypo1a3Lbbbf18mRQHVzqGqrIkUcemX/4h3/IQw89VOlRAKiADRs25LTTTtvr69OnT88vf/nLXpwIqov4gSpy4YUXZuXKlXnb296W8ePH55prrklra2ulxwKglzz99NNpbGzc6+uNjY3Ztm1bL04E1UX8QBWZO3du1q1bl1/96lc544wzct111+Xwww/PtGnTcuONN1Z6PAAOsF27dqW2du9f2e7fv39efPHFXpwIqovv/ECVu+uuu/KZz3wmGzZscLU3gDe4fv36Zfr06amrq3vF1zs7O7Ny5Up/D2AvXO0NqtT//M//ZNmyZfnOd76T9vb2fOhDH6r0SAAcYDNnzvyTa1zpDfbOzg9UkV//+tf5t3/7tyxbtiyPPPJITj755Jx77rk5++yzM2zYsEqPBwDQp4kfqCL9+vXLhAkTMmPGjHzkIx9JU1NTpUcCAKga4geqxK5du3L99dfnQx/6UPmmpwAAvHbiB6rIoEGD8uCDD2b06NGVHgUAoOq41DVUkXHjxuV3v/tdpccAAKhKdn6giqxatSqXXHJJ/umf/injx4/P0KFDu71eX19fockAAPo+8QNVpF+//9usrampKf+7VCqlpqbGfR0AAF6F+/xAFfnZz35W6REAAKqWnR8AAKAQ7PxAFbnjjjte9fUTTzyxlyYBAKg+dn6givzxd35e8sff/fGdHwCAvXOpa6gi27Zt6/bYunVrVq5cmXe9611ZtWpVpccDAOjT7PzAG8Add9yRv/u7v8v69esrPQoAQJ9l5wfeAEaOHJmHHnqo0mMAAPRpLngAVWTDhg3dnpdKpbS2tuaKK67IcccdV6GpAACqg4+9QRXp169fampq8vL/2p5wwgn55je/mbe+9a0VmgwAoO8TP1BFHn300W7P+/Xrl5EjR2bQoEEVmggAoHr4zg9Ugf/+7//Oj370oxxxxBHlx5o1a3LiiSfm8MMPz9/8zd+ks7Oz0mMCAPRp4geqwMKFC7t932fjxo2ZNWtWpkyZkksvvTQ//OEPs3jx4gpOCADQ9/nYG1SBQw89ND/84Q8zYcKEJMlll12WNWvWZO3atUmS7373u/nc5z6XBx54oJJjAgD0aXZ+oAps27YtjY2N5edr1qzJaaedVn7+rne9K5s3b67EaAAAVUP8QBVobGzMpk2bkiRdXV255557MnHixPLr27dvz4ABAyo1HgBAVRA/UAVOO+20XHrppfn5z3+eBQsWZMiQIXnPe95Tfn3Dhg056qijKjghAEDf5yanUAW+8IUv5Oyzz85JJ52Ugw46KDfccEMGDhxYfv2b3/xmpk2bVsEJAQD6Phc8gCrS3t6egw46KP379+92/Omnn85BBx3ULYgAAOhO/AAAAIXgOz8AAEAhiB8AAKAQxA8AAFAI4gcAACgE8QMAABSC+AEAAApB/AAAAIXw/wG7ZBLDpAMoIgAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 1000x500 with 1 Axes>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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AKAQxA8AAFAI4gcAACgE8QMAABSC+AEAAApB/AAAAIUgfgAAgEIQPwAAQCGIHwAAoBDEDwAAUAjiBwAAKATxAwAAFIL4AQAACkH8AAAAhSB+AACAQhA/AABAIYgfAACgEMQPAABQCOIHAAAoBPEDAAAUQnWlB6DzOviFxZUeASrukUoPAAC8bs78AAAAhSB+AACAQhA/AABAIYgfAACgEMQPAABQCOIHAAAoBPEDAAAUQrviZ968eXnPe96Tvn37ZtCgQTnllFPy4IMPtllTKpUyZ86cNDQ0pGfPnjnuuONy3333tVnT0tKSc889NwcccEB69+6dk08+ORs3bnzz7wYAAOBVtCt+Vq5cmXPOOSe33357VqxYkRdffDETJ07Mc889V15z+eWX54orrsiCBQuyevXq1NfXZ8KECdm6dWt5zYwZM7J06dIsWbIkq1atyrZt23LSSSdl586de+6dAQAA/JHq9ixetmxZm+ff+c53MmjQoKxZsybve9/7UiqVcuWVV+aiiy7KlClTkiSLFi1KXV1dFi9enLPPPjtNTU255pprct1112X8+PFJkuuvvz6DBw/OzTffnEmTJu2htwYAAPB/3tQ1P01NTUmS/v37J0nWr1+fxsbGTJw4sbympqYmY8eOzW233ZYkWbNmTXbs2NFmTUNDQ4YPH15e83ItLS1pbm5u8wAAAGiPNxw/pVIp559/fv7iL/4iw4cPT5I0NjYmSerq6tqsraurK+9rbGxM9+7ds//++7/qmpebN29eamtry4/Bgwe/0bEBAICCesPx87nPfS733HNPvve97+22r6qqqs3zUqm027aXe601s2fPTlNTU/mxYcOGNzo2AABQUG8ofs4999z8+Mc/zi9/+cu87W1vK2+vr69Pkt3O4GzevLl8Nqi+vj6tra3ZsmXLq655uZqamvTr16/NAwAAoD3aFT+lUimf+9zncsMNN+QXv/hFhg4d2mb/0KFDU19fnxUrVpS3tba2ZuXKlRkzZkySZOTIkenWrVubNZs2bcq9995bXgMAALCntetub+ecc04WL16c//zP/0zfvn3LZ3hqa2vTs2fPVFVVZcaMGZk7d26GDRuWYcOGZe7cuenVq1emTp1aXnvmmWdm5syZGTBgQPr3759Zs2ZlxIgR5bu/AQAA7Gntip+rr746SXLccce12f6d73wnp59+epLk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v36VHgeACvF5QNGVSqVs3bo1DQ0Nf3JtReLngAMOSNeuXXc7y7N58+bdzgYlSU1NTWpqatps22+//fbmiNBp9OvXz4cdAD4PKLQ/dcbnJRW54UH37t0zcuTIrFixos32FStWZMyYMZUYCQAAeIur2Nfezj///Hzyk5/MMccck9GjR+eb3/xmHnvssXzmM5+p1EgAAMBbWMXi5+Mf/3iefvrpXHrppdm0aVOGDx+eG2+8MQcddFClRoJOpaamJpdccsluXwkFoFh8HsDrV1V6PfeEAwAA6OQq9iOnAAAAHUn8AAAAhSB+AACAQhA/AABAIYgfAACgEMQPAABQCOIHAAAohIr9yCnw+kyZMuV1r73hhhv24iQAVNLXv/7117327/7u7/biJNB5iR/Yx9XW1pb/XSqVsnTp0tTW1uaYY45JkqxZsybPPvtsuyIJgM5n/vz5bZ4/+eSTef7557PffvslSZ599tn06tUrgwYNEj/wKsQP7OO+853vlP/9+c9/Pqeeemq+8Y1vpGvXrkmSnTt3Zvr06enXr1+lRgSgA6xfv77878WLF+eqq67KNddck3e84x1JkgcffDBnnXVWzj777EqNCPu8qlKpVKr0EMDrM3DgwKxatar8QfeSBx98MGPGjMnTTz9dockA6EiHHnpo/uM//iPvfve722xfs2ZNPvrRj7YJJeD/uOEBdCIvvvhi1q1bt9v2devWZdeuXRWYCIBK2LRpU3bs2LHb9p07d+aJJ56owETQOfjaG3Qin/rUp3LGGWfkd7/7XY499tgkye23357LLrssn/rUpyo8HQAdZdy4cTnrrLNyzTXXZOTIkamqqsodd9yRs88+O+PHj6/0eLDP8rU36ER27dqVr371q/na176WTZs2JUkOPPDAnHfeeZk5c2b5OiAA3tqefPLJTJs2LcuWLUu3bt2S/O+3AyZNmpSFCxdm0KBBFZ4Q9k3iBzqp5ubmJHGjA4AC++1vf5sHHnggpVIphx9+eA477LBKjwT7NPEDncyLL76YW265Jb///e8zderU9O3bN48//nj69euXPn36VHo8AIB9lviBTuTRRx/NCSeckMceeywtLS357W9/m0MOOSQzZszICy+8kG984xuVHhGADrJx48b8+Mc/zmOPPZbW1tY2+6644ooKTQX7Njc8gE7kvPPOyzHHHJO77747AwYMKG//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\n",
            "text/plain": [
              "<Figure size 1000x500 with 1 Axes>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAz8AAAHbCAYAAADlHyT+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAujUlEQVR4nO3df5SWdZ0//ufwa/jhMCuIM85xNDRSE/QUdBRO5i/EyB+1bmul21LZroa5ssAq5O7GugVqLdIun2hr3dBcwm1Xduu7RtAPIdZ1FzATlcw2FFhnRI1mQHFGh/v7R8e7RtQcYeae4Xo8zrnOmet9ve/7/bo81d3T9/V+X1WlUqkUAACAg1y/ShcAAADQE4QfAACgEIQfAACgEIQfAACgEIQfAACgEIQfAACgEIQfAACgEAZUuoA3Yu/evXniiSdSU1OTqqqqSpcDAABUSKlUyq5du9LQ0JB+/V57bqdPhp8nnngijY2NlS4DAADoJbZt25YjjzzyNfv0yfBTU1OT5Fc3OHz48ApXAwAAVEpra2saGxvLGeG19Mnw89KjbsOHDxd+AACA17UcxoYHAABAIQg/AABAIQg/AABAIfTJNT8AAMCv7d27N+3t7ZUuo9sMGjTot25j/XoIPwAA0Ie1t7dny5Yt2bt3b6VL6Tb9+vXL6NGjM2jQoP36HuEHAAD6qFKplKampvTv3z+NjY0HZHakt9m7d2+eeOKJNDU15aijjnpdu7q9GuEHAAD6qBdffDHPPfdcGhoaMnTo0EqX021GjRqVJ554Ii+++GIGDhz4hr/n4IuGAABQEB0dHUmy34+D9XYv3d9L9/tGCT8AANDH7c+jYH3Bgbo/4QcAACgE4QcAACgEGx4AAMBB5k1z/qNHx3vshvN6dLw3yswPAABQEV/84hczevToDB48OOPHj88Pf/jDbh1P+AEAAHrcHXfckRkzZuS6667Lj370o5x22mmZOnVqtm7d2m1jCj8AAECPW7hwYS677LJ8/OMfzwknnJBFixalsbExS5Ys6bYxhR8AAKBHtbe3Z+PGjZkyZUqn9ilTpuSee+7ptnFteAAA7Jftc7r3GX3o7Y684bRKl9DnPP300+no6EhdXV2n9rq6ujQ3N3fbuGZ+AACAinj5y0tLpVK3vrBV+AEAAHrUYYcdlv79++8zy7Njx459ZoMOJOEHAADoUYMGDcr48eOzevXqTu2rV6/OpEmTum1ca34AgP1yx5YbK10CVNSsWPPzRsycOTMf/vCHM2HChEycODFf/vKXs3Xr1lxxxRXdNqbwAwAAB5nHbjiv0iX8Vh/4wAfyzDPP5Prrr09TU1PGjh2bu+66K0cffXS3jSn8AAAAFTF9+vRMnz69x8az5gcAACgEMz8AwH4ZfOjMSpcA8LqY+QEAAApB+AEAAApB+AEAAApB+AEAAApB+AEAAApB+AEAAApB+AEAAArBe34AAOBgM6+2h8dr6fJH1q5dm8997nPZuHFjmpqasmLFirzvfe878LX9BjM/AABAj3v22Wdz8sknZ/HixT02ppkfAGC/nHX3lZUuASpsc6UL6JOmTp2aqVOn9uiYZn4AAIBCEH4AAIBCEH4AAIBCEH4AAIBCEH4AAIBCsNsbAADQ43bv3p2f/exn5fMtW7bk/vvvz4gRI3LUUUd1y5jCDwAA0OM2bNiQM888s3w+c+bMJMm0adOydOnSbhlT+AEA9svFc/3fCYptU6ULeCXzWipdwW91xhlnpFQq9eiY1vwAAACFIPwAAACF0KXwM2/evFRVVXU66uvry9dLpVLmzZuXhoaGDBkyJGeccUYeeuihTt/R1taWq666KocddliGDRuWCy+8MNu3bz8wdwMAAPAqujzzc+KJJ6apqal8bNr066ccb7rppixcuDCLFy/O+vXrU19fn3POOSe7du0q95kxY0ZWrFiR5cuXZ926ddm9e3fOP//8dHR0HJg7AgAAeAVdXqE4YMCATrM9LymVSlm0aFGuu+66XHTRRUmSW2+9NXV1dVm2bFkuv/zytLS05JZbbsnXvva1TJ48OUly++23p7GxMd/97ndz7rnnvuKYbW1taWtrK5+3trZ2tWwAAKDgujzz8+ijj6ahoSGjR4/OBz/4wfz85z9P8qt9uZubmzNlypRy3+rq6px++um55557kiQbN27MCy+80KlPQ0NDxo4dW+7zShYsWJDa2try0djY2NWyAQCAgutS+DnllFNy22235Tvf+U6+8pWvpLm5OZMmTcozzzyT5ubmJEldXV2nz9TV1ZWvNTc3Z9CgQTn00ENftc8rmTt3blpaWsrHtm3bulI2AABA1x57mzp1avnvcePGZeLEiTn22GNz66235tRTT02SVFVVdfpMqVTap+3lfluf6urqVFdXd6VUAACA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\n",
            "text/plain": [
              "<Figure size 1000x500 with 1 Axes>"
            ]
          },
          "metadata": {}
        }
      ],
      "id": "4abcda4c"
    },
    {
      "cell_type": "markdown",
      "source": "일단 성별, 사회적 지위, 탑승지역, 타이타닉에 탑승한 형제,자매 수를 시각화 하였다. 그 결과 여성보다 남성이 더 많이 생존했고, 지위가 높거나 가족이 있는 승객이 더 높은 생존율을 보이고 있다.",
      "metadata": {
        "papermill": {
          "duration": 0.016971,
          "end_time": "2023-04-09T14:23:35.656030",
          "exception": false,
          "start_time": "2023-04-09T14:23:35.639059",
          "status": "completed"
        },
        "tags": []
      },
      "id": "213dc749"
    },
    {
      "cell_type": "code",
      "source": "train_test_data = [train, test]\n\nsex_map = {\"male\":0, \"female\":1}\nfor dataset in train_test_data:\n    dataset['Sex'] = dataset['Sex'].map(sex_map)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:35.692790Z",
          "iopub.status.busy": "2023-04-09T14:23:35.692308Z",
          "iopub.status.idle": "2023-04-09T14:23:35.701717Z",
          "shell.execute_reply": "2023-04-09T14:23:35.700434Z"
        },
        "papermill": {
          "duration": 0.030668,
          "end_time": "2023-04-09T14:23:35.704113",
          "exception": false,
          "start_time": "2023-04-09T14:23:35.673445",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 14,
      "outputs": [],
      "id": "cab48753"
    },
    {
      "cell_type": "markdown",
      "source": "훈련데이터와 테스트 데이터를 한번에 전처리 하기 위해서 하나의 리스트로 묶어서 한번에 처리하였다.",
      "metadata": {
        "papermill": {
          "duration": 0.017234,
          "end_time": "2023-04-09T14:23:35.738665",
          "exception": false,
          "start_time": "2023-04-09T14:23:35.721431",
          "status": "completed"
        },
        "tags": []
      },
      "id": "78406eeb"
    },
    {
      "cell_type": "code",
      "source": "train.head()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:35.777204Z",
          "iopub.status.busy": "2023-04-09T14:23:35.776028Z",
          "iopub.status.idle": "2023-04-09T14:23:35.794880Z",
          "shell.execute_reply": "2023-04-09T14:23:35.793551Z"
        },
        "papermill": {
          "duration": 0.041197,
          "end_time": "2023-04-09T14:23:35.797582",
          "exception": false,
          "start_time": "2023-04-09T14:23:35.756385",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 15,
      "outputs": [
        {
          "execution_count": 15,
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>1</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>1</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>1</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>0</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name  Sex   Age  SibSp  Parch  \\\n",
              "0                            Braund, Mr. Owen Harris    0  22.0      1      0   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    1  38.0      1      0   \n",
              "2                             Heikkinen, Miss. Laina    1  26.0      0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    1  35.0      1      0   \n",
              "4                           Allen, Mr. William Henry    0  35.0      0      0   \n",
              "\n",
              "             Ticket     Fare Cabin Embarked  \n",
              "0         A/5 21171   7.2500   NaN        S  \n",
              "1          PC 17599  71.2833   C85        C  \n",
              "2  STON/O2. 3101282   7.9250   NaN        S  \n",
              "3            113803  53.1000  C123        S  \n",
              "4            373450   8.0500   NaN        S  "
            ]
          },
          "metadata": {}
        }
      ],
      "id": "727d42d8"
    },
    {
      "cell_type": "code",
      "source": "for dataset in train_test_data:\n    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인\n    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1\n    dataset['Alone'] = 1\n    \n    # 가족수 > 1이면 동승자 있음\n    dataset.loc[dataset['Family'] > 1, 'Alone'] = 0",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:35.835715Z",
          "iopub.status.busy": "2023-04-09T14:23:35.835232Z",
          "iopub.status.idle": "2023-04-09T14:23:35.846394Z",
          "shell.execute_reply": "2023-04-09T14:23:35.845378Z"
        },
        "papermill": {
          "duration": 0.033328,
          "end_time": "2023-04-09T14:23:35.848692",
          "exception": false,
          "start_time": "2023-04-09T14:23:35.815364",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 16,
      "outputs": [],
      "id": "4abc5c6f"
    },
    {
      "cell_type": "markdown",
      "source": "SibSp & Parch 항목은 결국 같이 탑승한 가족의 수로 볼 수 있으므로 하나로 묶어주었다. Family 항목으로 묶고 아닐경우 Alone으로 묶어보았다.",
      "metadata": {
        "papermill": {
          "duration": 0.017547,
          "end_time": "2023-04-09T14:23:35.884334",
          "exception": false,
          "start_time": "2023-04-09T14:23:35.866787",
          "status": "completed"
        },
        "tags": []
      },
      "id": "1d15c085"
    },
    {
      "cell_type": "code",
      "source": "chart('Alone')\n",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:35.922649Z",
          "iopub.status.busy": "2023-04-09T14:23:35.922181Z",
          "iopub.status.idle": "2023-04-09T14:23:36.164721Z",
          "shell.execute_reply": "2023-04-09T14:23:36.163140Z"
        },
        "papermill": {
          "duration": 0.26522,
          "end_time": "2023-04-09T14:23:36.168127",
          "exception": false,
          "start_time": "2023-04-09T14:23:35.902907",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1000x500 with 1 Axes>"
            ]
          },
          "metadata": {}
        }
      ],
      "id": "fe90db9e"
    },
    {
      "cell_type": "markdown",
      "source": "확인 결과 혼자 탄 사람이 더 많이 사망한 것을 확인했다.",
      "metadata": {
        "papermill": {
          "duration": 0.018971,
          "end_time": "2023-04-09T14:23:36.206039",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.187068",
          "status": "completed"
        },
        "tags": []
      },
      "id": "d3e92c04"
    },
    {
      "cell_type": "code",
      "source": "train.head()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:36.245421Z",
          "iopub.status.busy": "2023-04-09T14:23:36.244720Z",
          "iopub.status.idle": "2023-04-09T14:23:36.266849Z",
          "shell.execute_reply": "2023-04-09T14:23:36.265348Z"
        },
        "papermill": {
          "duration": 0.045526,
          "end_time": "2023-04-09T14:23:36.270230",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.224704",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 18,
      "outputs": [
        {
          "execution_count": 18,
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "      <th>Family</th>\n",
              "      <th>Alone</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>1</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>1</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>1</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>0</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name  Sex   Age  SibSp  Parch  \\\n",
              "0                            Braund, Mr. Owen Harris    0  22.0      1      0   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    1  38.0      1      0   \n",
              "2                             Heikkinen, Miss. Laina    1  26.0      0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    1  35.0      1      0   \n",
              "4                           Allen, Mr. William Henry    0  35.0      0      0   \n",
              "\n",
              "             Ticket     Fare Cabin Embarked  Family  Alone  \n",
              "0         A/5 21171   7.2500   NaN        S       2      0  \n",
              "1          PC 17599  71.2833   C85        C       2      0  \n",
              "2  STON/O2. 3101282   7.9250   NaN        S       1      1  \n",
              "3            113803  53.1000  C123        S       2      0  \n",
              "4            373450   8.0500   NaN        S       1      1  "
            ]
          },
          "metadata": {}
        }
      ],
      "id": "8f330d18"
    },
    {
      "cell_type": "code",
      "source": "class_list=[]\nfor i in range(1,4):\n    series = train[train['Pclass'] == i]['Embarked'].value_counts()\n    class_list.append(series)\n\ndf = pd.DataFrame(class_list)\ndf.index = ['1st', '2nd', '3rd']\ndf.plot(kind=\"bar\", figsize=(10,5))",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:36.316398Z",
          "iopub.status.busy": "2023-04-09T14:23:36.314867Z",
          "iopub.status.idle": "2023-04-09T14:23:36.583507Z",
          "shell.execute_reply": "2023-04-09T14:23:36.582202Z"
        },
        "papermill": {
          "duration": 0.292245,
          "end_time": "2023-04-09T14:23:36.586857",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.294612",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 19,
      "outputs": [
        {
          "execution_count": 19,
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<AxesSubplot:>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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iLf8x4/XvPzyy708Df2V+KFfWLRoUZKkVCrlmmuuyZAhQyrH9u3blx//+Mf5j//xP1ZpOqC3jRs3rvLnP/3TP83Xv/71fOQjH6nsmzx5clpaWnLNNdfk4x//eBUmBHrbzTffXPnztm3b8tWvfjVnn312t38EWbt2ba655poqTUh/4G1v9AtnnHFGkmT9+vWZOnVqampqKsdqampy7LHHZsmSJZkwYUK1RgT6yODBg/OTn/wkJ5xwQrf9Tz/9dE455ZTs3r27SpMBfWXOnDk544wzcsUVV3Tbv2LFijz00EO57777qjMY73rih37lU5/6VL72ta+lrq6u2qMAVXLKKafkhBNOyG233VZ5u2tXV1c+/elP5+mnn85PfvKTKk8I9LZhw4bliSeeyHHHHddt/7PPPpuTTz45O3furNJkvNt52xv9yu23397tcWdnZx5++OG8733vy/ve974qTQX0pW9+85s577zz0tLSkpNOOilJ8rOf/SylUin/83/+zypPB/SFUaNGZfXq1fnc5z7Xbf99992XUaNGVWkq+gNXfuhX5s6dmw9/+MO54oorsnv37px00kl5/vnnUy6Xs2rVqsyZM6faIwJ94NVXX83dd9+dZ555JuVyOSeeeGIuvvjiDB06tNqjAX1g5cqVmT9/fs4555zKZ34ee+yxrFmzJv/0T/+UefPmVXdA3rXED/1KY2Nj1q5dm5NOOin33HNPvvzlL+dnP/tZ7rjjjvzDP/xDfvrTn1Z7RACgD/z4xz/O17/+9Tz99NOVfwT5q7/6q5x22mnVHo13MfFDvzJ48OD86le/SktLS/7iL/4izc3Nue666/LCCy/kxBNP9B5fKIhf/epXeeSRR9Le3p79+/d3O/alL32pSlMBfWHv3r35L//lv+Saa67Je97znmqPQz/jMz/0Ky0tLfnRj36UkSNHZs2aNVm1alWSZPv27b7nBwriH//xH/PZz342o0ePTmNjY7fv/SiVSuIHDnODBg3K6tWr3dKagyJ+6FcWLlyYP//zP8+wYcMybty4TJ8+PUnywx/+MJMmTarucECf+OpXv5prr702n//856s9ClAlF1xwQe67777K9wDC2+Vtb/Q7mzZtygsvvJCZM2dm2LBhSZIHHnggI0aMyLRp06o8HdDb6urq8sQTT3i7CxTYtddem//23/5bzjrrrEyZMuV1Nzv5q7/6qypNxrud+OGwsHXr1nz5y1/Of//v/73aowC9bP78+fnABz6Qz3zmM9UeBaiS8ePHH/BYqVTKr3/96z6chv5E/HBY+NnPfpZTTjkl+/btq/YoQC9bvnx5brzxxnz0ox/NpEmTMmjQoG7H/YsvAAcifugX7r///jc9/utf/zqLFy8WP1AA/sUXiu3pp5/OY489lmnTpuW9731vnnnmmXzta19LV1dX/tN/+k8588wzqz0i72Lih37hiCOOSKlUypv9dS2VSuIHAA5ja9asycc+9rEMGzYsr776alavXp2/+Iu/yEknnZRyuZz169dn7dq1AogDOqLaA8Db0dTUlO985zvZv3//G24/+clPqj0iANDL/vqv/zqf+9znsm3bttx+++25+OKLc+mll2bdunV56KGHcuWVV+a6666r9pi8i4kf+oUpU6a8aeC81VUh4PCwe/fuPProo/nFL37xumP/9m//ljvvvLMKUwF95amnnsq8efOSJHPnzs2OHTsyZ86cyvFPfOIT+fnPf16l6egPxA/9wuc+97k3vY31cccdlx/84Ad9OBHQ1371q1/lhBNOyIc//OFMmjQp06dPT2tra+V4R0dHPvWpT1VxQqAvHXHEETnyyCNz1FFHVfYNHz48HR0d1RuKdz3xQ7/wJ3/yJznnnHMOeHzo0KE5/fTT+3AioK99/vOfz6RJk9Le3p5f/vKXqaury4c+9KG88MIL1R4N6CPHHntsnnvuucrjH/3oRxk7dmzl8datW9PU1FSN0egnxA8A/cKGDRuybNmyjB49Oscdd1zuv//+zJ49O3/yJ3/iDm9QEJ/97Ge73dxo4sSJGThwYOXx9773PTc74E252xsA/UJdXV1+/OMf54QTTui2f8GCBbnvvvtyzz33ZPr06e76CMABDXzrJQBQfe973/vy+OOPvy5+brnllpTL5Zx//vlVmgyA/sLb3gDoFy644IJ8+9vffsNjK1asyCc+8Ql3fQTgTXnbGwAAUAiu/AAAAIUgfgAAgEIQPwAAQCGIHwAAoBDEDwAAUAjiBwAAKATxAwAAFIL4AQAACuH/AfY5x3+jmisXAAAAAElFTkSuQmCC\n",
            "text/plain": [
              "<Figure size 1000x500 with 1 Axes>"
            ]
          },
          "metadata": {}
        }
      ],
      "id": "7b33252a"
    },
    {
      "cell_type": "markdown",
      "source": "막대 그래프를 보면 낮은 지위의 사람들이 더 낮은 생존율을 보였다. 이를 확실히 확인해보기 위해 거주 지역의 차이를 시각화하였다. 그 결과 Q지역이 상대적으로 빈곤한 지역으로 보였다.",
      "metadata": {
        "papermill": {
          "duration": 0.018366,
          "end_time": "2023-04-09T14:23:36.629874",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.611508",
          "status": "completed"
        },
        "tags": []
      },
      "id": "26d1e095"
    },
    {
      "cell_type": "code",
      "source": "for dataset in train_test_data:\n    dataset['Embarked'] = dataset['Embarked'].fillna('S')",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:36.669981Z",
          "iopub.status.busy": "2023-04-09T14:23:36.668840Z",
          "iopub.status.idle": "2023-04-09T14:23:36.677472Z",
          "shell.execute_reply": "2023-04-09T14:23:36.676226Z"
        },
        "papermill": {
          "duration": 0.031616,
          "end_time": "2023-04-09T14:23:36.680137",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.648521",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 20,
      "outputs": [],
      "id": "a0ea6d4c"
    },
    {
      "cell_type": "markdown",
      "source": "확인 결과 대부분의 사람들이 S지역에서 탑승했으므로 결측치는 S로 채워줬다.",
      "metadata": {
        "papermill": {
          "duration": 0.018384,
          "end_time": "2023-04-09T14:23:36.717187",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.698803",
          "status": "completed"
        },
        "tags": []
      },
      "id": "3a9eb1c5"
    },
    {
      "cell_type": "code",
      "source": "embarked_mapping = {'S':0, 'C':1, 'Q':2}\nfor dataset in train_test_data:\n    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)\n    \ntrain.head()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:36.756993Z",
          "iopub.status.busy": "2023-04-09T14:23:36.755937Z",
          "iopub.status.idle": "2023-04-09T14:23:36.776333Z",
          "shell.execute_reply": "2023-04-09T14:23:36.775160Z"
        },
        "papermill": {
          "duration": 0.04394,
          "end_time": "2023-04-09T14:23:36.779831",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.735891",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 21,
      "outputs": [
        {
          "execution_count": 21,
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "      <th>Family</th>\n",
              "      <th>Alone</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>1</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>1</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>1</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>0</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name  Sex   Age  SibSp  Parch  \\\n",
              "0                            Braund, Mr. Owen Harris    0  22.0      1      0   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    1  38.0      1      0   \n",
              "2                             Heikkinen, Miss. Laina    1  26.0      0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    1  35.0      1      0   \n",
              "4                           Allen, Mr. William Henry    0  35.0      0      0   \n",
              "\n",
              "             Ticket     Fare Cabin  Embarked  Family  Alone  \n",
              "0         A/5 21171   7.2500   NaN         0       2      0  \n",
              "1          PC 17599  71.2833   C85         1       2      0  \n",
              "2  STON/O2. 3101282   7.9250   NaN         0       1      1  \n",
              "3            113803  53.1000  C123         0       2      0  \n",
              "4            373450   8.0500   NaN         0       1      1  "
            ]
          },
          "metadata": {}
        }
      ],
      "id": "1a7ad0af"
    },
    {
      "cell_type": "markdown",
      "source": "탑승지역을 숫자로 치환해서 입력",
      "metadata": {
        "papermill": {
          "duration": 0.022352,
          "end_time": "2023-04-09T14:23:36.831045",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.808693",
          "status": "completed"
        },
        "tags": []
      },
      "id": "5b96a0b0"
    },
    {
      "cell_type": "code",
      "source": "for dataset in train_test_data:\n    dataset['Marry'] = dataset['Name'].str.extract('([\\w]+)\\.', expand=False)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:36.872016Z",
          "iopub.status.busy": "2023-04-09T14:23:36.870934Z",
          "iopub.status.idle": "2023-04-09T14:23:36.884854Z",
          "shell.execute_reply": "2023-04-09T14:23:36.883517Z"
        },
        "papermill": {
          "duration": 0.037509,
          "end_time": "2023-04-09T14:23:36.887541",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.850032",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 22,
      "outputs": [],
      "id": "87fefceb"
    },
    {
      "cell_type": "markdown",
      "source": "이름은 항목에 있지만 이름이 생존율에 유의미한 영향을 미쳤다고 생각하기는 어렵다. 그러나 이름에 무엇이 붙었냐에 따라 결혼/미혼 여부 판단이 가능할 것으로 판단된다. 따라서 이름 옆에 특정 부분을 추출하여 Marry에 저장했다.",
      "metadata": {
        "papermill": {
          "duration": 0.018402,
          "end_time": "2023-04-09T14:23:36.924894",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.906492",
          "status": "completed"
        },
        "tags": []
      },
      "id": "5db86cde"
    },
    {
      "cell_type": "code",
      "source": "train.head()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:36.965623Z",
          "iopub.status.busy": "2023-04-09T14:23:36.964595Z",
          "iopub.status.idle": "2023-04-09T14:23:36.987641Z",
          "shell.execute_reply": "2023-04-09T14:23:36.986349Z"
        },
        "papermill": {
          "duration": 0.046538,
          "end_time": "2023-04-09T14:23:36.990307",
          "exception": false,
          "start_time": "2023-04-09T14:23:36.943769",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 23,
      "outputs": [
        {
          "execution_count": 23,
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "      <th>Family</th>\n",
              "      <th>Alone</th>\n",
              "      <th>Marry</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>Mr</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>1</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>Mrs</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>1</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Miss</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>1</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>Mrs</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>0</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Mr</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name  Sex   Age  SibSp  Parch  \\\n",
              "0                            Braund, Mr. Owen Harris    0  22.0      1      0   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    1  38.0      1      0   \n",
              "2                             Heikkinen, Miss. Laina    1  26.0      0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    1  35.0      1      0   \n",
              "4                           Allen, Mr. William Henry    0  35.0      0      0   \n",
              "\n",
              "             Ticket     Fare Cabin  Embarked  Family  Alone Marry  \n",
              "0         A/5 21171   7.2500   NaN         0       2      0    Mr  \n",
              "1          PC 17599  71.2833   C85         1       2      0   Mrs  \n",
              "2  STON/O2. 3101282   7.9250   NaN         0       1      1  Miss  \n",
              "3            113803  53.1000  C123         0       2      0   Mrs  \n",
              "4            373450   8.0500   NaN         0       1      1    Mr  "
            ]
          },
          "metadata": {}
        }
      ],
      "id": "91eb0cae"
    },
    {
      "cell_type": "code",
      "source": "train['Marry'].value_counts()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:37.032695Z",
          "iopub.status.busy": "2023-04-09T14:23:37.031909Z",
          "iopub.status.idle": "2023-04-09T14:23:37.041231Z",
          "shell.execute_reply": "2023-04-09T14:23:37.039887Z"
        },
        "papermill": {
          "duration": 0.033677,
          "end_time": "2023-04-09T14:23:37.043576",
          "exception": false,
          "start_time": "2023-04-09T14:23:37.009899",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 24,
      "outputs": [
        {
          "execution_count": 24,
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Mr          517\n",
              "Miss        182\n",
              "Mrs         125\n",
              "Master       40\n",
              "Dr            7\n",
              "Rev           6\n",
              "Mlle          2\n",
              "Major         2\n",
              "Col           2\n",
              "Countess      1\n",
              "Capt          1\n",
              "Ms            1\n",
              "Sir           1\n",
              "Lady          1\n",
              "Mme           1\n",
              "Don           1\n",
              "Jonkheer      1\n",
              "Name: Marry, dtype: int64"
            ]
          },
          "metadata": {}
        }
      ],
      "id": "aa24b34a"
    },
    {
      "cell_type": "code",
      "source": "test['Marry'].value_counts()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:37.084950Z",
          "iopub.status.busy": "2023-04-09T14:23:37.084458Z",
          "iopub.status.idle": "2023-04-09T14:23:37.095496Z",
          "shell.execute_reply": "2023-04-09T14:23:37.094189Z"
        },
        "papermill": {
          "duration": 0.034911,
          "end_time": "2023-04-09T14:23:37.097898",
          "exception": false,
          "start_time": "2023-04-09T14:23:37.062987",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 25,
      "outputs": [
        {
          "execution_count": 25,
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Mr        240\n",
              "Miss       78\n",
              "Mrs        72\n",
              "Master     21\n",
              "Col         2\n",
              "Rev         2\n",
              "Ms          1\n",
              "Dr          1\n",
              "Dona        1\n",
              "Name: Marry, dtype: int64"
            ]
          },
          "metadata": {}
        }
      ],
      "id": "343544cf"
    },
    {
      "cell_type": "markdown",
      "source": "value_counts() 함수를 사용하여 호칭을 분류하여 그 숫자를 카운트해보았다.",
      "metadata": {
        "papermill": {
          "duration": 0.01916,
          "end_time": "2023-04-09T14:23:37.136601",
          "exception": false,
          "start_time": "2023-04-09T14:23:37.117441",
          "status": "completed"
        },
        "tags": []
      },
      "id": "9c2f2f93"
    },
    {
      "cell_type": "code",
      "source": "for dataset in train_test_data:\n    dataset['Marry'] = dataset['Marry'].apply(lambda x: 0 if x==\"Mr\" else 1 if x==\"Miss\" else 2 if x==\"Mrs\" else 3 if x==\"Master\" else 4)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:37.178183Z",
          "iopub.status.busy": "2023-04-09T14:23:37.177412Z",
          "iopub.status.idle": "2023-04-09T14:23:37.184679Z",
          "shell.execute_reply": "2023-04-09T14:23:37.183740Z"
        },
        "papermill": {
          "duration": 0.031168,
          "end_time": "2023-04-09T14:23:37.187084",
          "exception": false,
          "start_time": "2023-04-09T14:23:37.155916",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 26,
      "outputs": [],
      "id": "9a196482"
    },
    {
      "cell_type": "markdown",
      "source": "서양에서 주로 사용하는 mr,miss,mrs,master를 제외한 다른 항목들은 하나로 묶어서 취급했다.",
      "metadata": {
        "papermill": {
          "duration": 0.01897,
          "end_time": "2023-04-09T14:23:37.225340",
          "exception": false,
          "start_time": "2023-04-09T14:23:37.206370",
          "status": "completed"
        },
        "tags": []
      },
      "id": "9bff0d7e"
    },
    {
      "cell_type": "code",
      "source": "train['Marry'].value_counts()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:37.266062Z",
          "iopub.status.busy": "2023-04-09T14:23:37.265578Z",
          "iopub.status.idle": "2023-04-09T14:23:37.274089Z",
          "shell.execute_reply": "2023-04-09T14:23:37.273020Z"
        },
        "papermill": {
          "duration": 0.031705,
          "end_time": "2023-04-09T14:23:37.276324",
          "exception": false,
          "start_time": "2023-04-09T14:23:37.244619",
          "status": "completed"
        },
        "tags": []
      },
      "execution_count": 27,
      "outputs": [
        {
          "execution_count": 27,
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    517\n",
              "1    182\n",
              "2    125\n",
              "3     40\n",
              "4     27\n",
              "Name: Marry, dtype: int64"
            ]
          },
          "metadata": {}
        }
      ],
      "id": "72ab311f"
    },
    {
      "cell_type": "code",
      "source": "test['Marrye'].value_counts()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T14:23:37.318057Z",
          "iopub.status.busy": "2023-04-09T14:23:37.317160Z",
          "iopub.status.idle": "2023-04-09T14:23:37.487379Z",
          "shell.execute_reply": "2023-04-09T14:23:37.485389Z"
        },
        "papermill": {
          "duration": 0.193806,
          "end_time": "2023-04-09T14:23:37.489943",
          "exception": true,
          "start_time": "2023-04-09T14:23:37.296137",
          "status": "failed"
        },
        "tags": []
      },
      "execution_count": 28,
      "outputs": [
        {
          "ename": "KeyError",
          "evalue": "'Marrye'",
          "output_type": "error",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
            "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/pandas/core/indexes/base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   3360\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3361\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3362\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/pandas/_libs/index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[0;34m()\u001b[0m\n",
            "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/pandas/_libs/index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[0;34m()\u001b[0m\n",
            "\u001b[0;32mpandas/_libs/hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[0;34m()\u001b[0m\n",
            "\u001b[0;32mpandas/_libs/hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[0;34m()\u001b[0m\n",
            "\u001b[0;31mKeyError\u001b[0m: 'Marrye'",
            "\nThe above exception was the direct cause of the following exception:\n",
            "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_19/2940886444.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mtest\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'Marrye'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalue_counts\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   3456\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnlevels\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3457\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3458\u001b[0;31m             \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3459\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mis_integer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3460\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/pandas/core/indexes/base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   3361\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3362\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3363\u001b[0;31m                 \u001b[0;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3364\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3365\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mis_scalar\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0misna\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhasnans\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyError\u001b[0m: 'Marrye'"
          ]
        }
      ],
      "id": "30828fcc"
    },
    {
      "cell_type": "code",
      "source": "chart('Marry')",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:07:18.726411Z",
          "iopub.status.busy": "2023-04-09T11:07:18.725747Z",
          "iopub.status.idle": "2023-04-09T11:07:19.002383Z",
          "shell.execute_reply": "2023-04-09T11:07:19.001020Z",
          "shell.execute_reply.started": "2023-04-09T11:07:18.726365Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "471e5e79"
    },
    {
      "cell_type": "markdown",
      "source": "이후 막대 그래프를 확인해보니 mr의 사망율이 압도적으로 높았고 mrs,miss가 생존율이 높은 반면 가족이 없는 miss의 경우 상대적으로 낮은 것을 확인했다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "85bb1508"
    },
    {
      "cell_type": "code",
      "source": "train['Cabin'].value_counts()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:08:50.725841Z",
          "iopub.status.busy": "2023-04-09T11:08:50.725433Z",
          "iopub.status.idle": "2023-04-09T11:08:50.737479Z",
          "shell.execute_reply": "2023-04-09T11:08:50.736533Z",
          "shell.execute_reply.started": "2023-04-09T11:08:50.725807Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "36597101"
    },
    {
      "cell_type": "code",
      "source": "train['Cabin'] = train['Cabin'].str[:1]",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:08:59.959008Z",
          "iopub.status.busy": "2023-04-09T11:08:59.958211Z",
          "iopub.status.idle": "2023-04-09T11:08:59.966072Z",
          "shell.execute_reply": "2023-04-09T11:08:59.964689Z",
          "shell.execute_reply.started": "2023-04-09T11:08:59.958957Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "37922db9"
    },
    {
      "cell_type": "markdown",
      "source": "cabin은 방번호 임으로 숫자를 제외하고 알파벳만 추출한다. ",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "e887564e"
    },
    {
      "cell_type": "code",
      "source": "class_list=[]\nfor i in range(1,4):\n    a = train[train['Pclass'] == i]['Cabin'].value_counts()\n    class_list.append(a)\n\ndf = pd.DataFrame(class_list)\ndf.index = ['1st', '2nd', '3rd']\ndf.plot(kind=\"bar\", figsize=(10,5))",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:09:09.757981Z",
          "iopub.status.busy": "2023-04-09T11:09:09.757559Z",
          "iopub.status.idle": "2023-04-09T11:09:10.098248Z",
          "shell.execute_reply": "2023-04-09T11:09:10.097063Z",
          "shell.execute_reply.started": "2023-04-09T11:09:09.757941Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "bee68926"
    },
    {
      "cell_type": "markdown",
      "source": "막대 그래프 확인결과 1등급의 생존율이 가장 높고 그다음 2등급, 3등급이 가장 낮았다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "11c19ed8"
    },
    {
      "cell_type": "code",
      "source": "for dataset in train_test_data:\n    dataset['Age'].fillna(dataset.groupby(\"Title\")[\"Age\"].transform(\"median\"), inplace=True)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:09:21.771363Z",
          "iopub.status.busy": "2023-04-09T11:09:21.770959Z",
          "iopub.status.idle": "2023-04-09T11:09:21.784296Z",
          "shell.execute_reply": "2023-04-09T11:09:21.782820Z",
          "shell.execute_reply.started": "2023-04-09T11:09:21.771327Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "42a130ed"
    },
    {
      "cell_type": "markdown",
      "source": "나이에 대한 정보에는 누락된 값이 존재했다. 따라서 Marry에 해당하는 그룹의 가운데값을 결측치로 바꿔준다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "28c66b22"
    },
    {
      "cell_type": "code",
      "source": "g = sns.FacetGrid(train, hue=\"Survived\", aspect=4)\ng = (g.map(sns.kdeplot, \"Age\").add_legend())",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:10:09.174083Z",
          "iopub.status.busy": "2023-04-09T11:10:09.172722Z",
          "iopub.status.idle": "2023-04-09T11:10:09.581047Z",
          "shell.execute_reply": "2023-04-09T11:10:09.579807Z",
          "shell.execute_reply.started": "2023-04-09T11:10:09.174010Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "5ab57f98"
    },
    {
      "cell_type": "markdown",
      "source": "나이대별 생존율을 비교하기 위해 선 그래프를 이용해보았다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "ace7b2e9"
    },
    {
      "cell_type": "code",
      "source": "for dataset in train_test_data:\n    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:11:21.415261Z",
          "iopub.status.busy": "2023-04-09T11:11:21.414772Z",
          "iopub.status.idle": "2023-04-09T11:11:21.428369Z",
          "shell.execute_reply": "2023-04-09T11:11:21.427130Z",
          "shell.execute_reply.started": "2023-04-09T11:11:21.415224Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "c7762abe"
    },
    {
      "cell_type": "code",
      "source": "for dataset in train_test_data:\n    dataset.loc[dataset['Age'] <= 15, 'Age'] = 0\n    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 25), 'Age'] = 1\n    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 35), 'Age'] = 2\n    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 3\n    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 60), 'Age'] = 4\n    dataset.loc[dataset['Age'] > 60, 'Age'] = 5",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "3f66a201"
    },
    {
      "cell_type": "markdown",
      "source": "각 데이터를 더 잘 비교하기 위해 나잇대의 그룹을 더 세밀하게 나눠보았다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "a9d336fb"
    },
    {
      "cell_type": "code",
      "source": "for dataset in train_test_data:\n    dataset[\"Fare\"].fillna(dataset.groupby(\"Pclass\")[\"Fare\"].transform(\"median\"), inplace=True)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:13:10.475162Z",
          "iopub.status.busy": "2023-04-09T11:13:10.473857Z",
          "iopub.status.idle": "2023-04-09T11:13:10.485110Z",
          "shell.execute_reply": "2023-04-09T11:13:10.483571Z",
          "shell.execute_reply.started": "2023-04-09T11:13:10.475101Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "1eaeb0d2"
    },
    {
      "cell_type": "markdown",
      "source": "Fare은 탑승요금인데 높은 등급일수록 높다. 따라서 Fare에서 누락된 값은 Pclass의 중간값을 사용해준다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "473436f2"
    },
    {
      "cell_type": "code",
      "source": "g = sns.FacetGrid(train, hue=\"Survived\", aspect=4)\ng = (g.map(sns.kdeplot, \"Fare\")\n     .add_legend()\n     .set(xlim=(0, train['Fare'].max()))) # x축 범위 설정",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:13:17.531394Z",
          "iopub.status.busy": "2023-04-09T11:13:17.530985Z",
          "iopub.status.idle": "2023-04-09T11:13:18.115448Z",
          "shell.execute_reply": "2023-04-09T11:13:18.114147Z",
          "shell.execute_reply.started": "2023-04-09T11:13:17.531360Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "78b9dad7"
    },
    {
      "cell_type": "markdown",
      "source": "승객별로 탑승요금의 편차가 굉장히 크고 분포는 우측 꼬리가 길게 편향되어 있다. 따라서 데이터를 그룹화할때 길이가 아닌 개수를 기준으로 나눈다음 Farebin이라는 열에 저장한다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "3f258157"
    },
    {
      "cell_type": "code",
      "source": "for dataset in train_test_data:\n    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:13:24.746034Z",
          "iopub.status.busy": "2023-04-09T11:13:24.745256Z",
          "iopub.status.idle": "2023-04-09T11:13:24.756224Z",
          "shell.execute_reply": "2023-04-09T11:13:24.754970Z",
          "shell.execute_reply.started": "2023-04-09T11:13:24.745988Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "d0b718c4"
    },
    {
      "cell_type": "code",
      "source": "pd.qcut(train['Fare'], 4)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:13:29.590916Z",
          "iopub.status.busy": "2023-04-09T11:13:29.589983Z",
          "iopub.status.idle": "2023-04-09T11:13:29.611462Z",
          "shell.execute_reply": "2023-04-09T11:13:29.610235Z",
          "shell.execute_reply.started": "2023-04-09T11:13:29.590873Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "1c4fc433"
    },
    {
      "cell_type": "code",
      "source": "drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']\n\nfor dataset in train_test_data:\n    dataset = dataset.drop(drop_column, axis=1, inplace=True)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:13:47.534143Z",
          "iopub.status.busy": "2023-04-09T11:13:47.533677Z",
          "iopub.status.idle": "2023-04-09T11:13:47.542526Z",
          "shell.execute_reply": "2023-04-09T11:13:47.541569Z",
          "shell.execute_reply.started": "2023-04-09T11:13:47.534105Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "e615db1e"
    },
    {
      "cell_type": "markdown",
      "source": "이제 훈련에서 제외할 항목들을 삭제해준다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "d325d4b7"
    },
    {
      "cell_type": "code",
      "source": "train.head()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:14:45.581040Z",
          "iopub.status.busy": "2023-04-09T11:14:45.580477Z",
          "iopub.status.idle": "2023-04-09T11:14:45.603019Z",
          "shell.execute_reply": "2023-04-09T11:14:45.601225Z",
          "shell.execute_reply.started": "2023-04-09T11:14:45.580987Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "bcbbecfa"
    },
    {
      "cell_type": "code",
      "source": "drop_column2 = ['PassengerId', 'Survived']\ntrain_data = train.drop(drop_column2, axis=1)\ntarget = train['Survived']",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:15:40.439573Z",
          "iopub.status.busy": "2023-04-09T11:15:40.439146Z",
          "iopub.status.idle": "2023-04-09T11:15:40.446984Z",
          "shell.execute_reply": "2023-04-09T11:15:40.445404Z",
          "shell.execute_reply.started": "2023-04-09T11:15:40.439532Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "e1a537fd"
    },
    {
      "cell_type": "markdown",
      "source": "PassengerId는 필요없으므로 삭제해주고 survived는 결과값임으로 마찬가지로 삭제해준다. 이 둘은 따로 target에 저장후 훈련데이터에서는 삭제해주었다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "15c099aa"
    },
    {
      "cell_type": "code",
      "source": "from sklearn.linear_model import LogisticRegression",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:15:57.092515Z",
          "iopub.status.busy": "2023-04-09T11:15:57.092094Z",
          "iopub.status.idle": "2023-04-09T11:15:57.098203Z",
          "shell.execute_reply": "2023-04-09T11:15:57.096949Z",
          "shell.execute_reply.started": "2023-04-09T11:15:57.092474Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "f45bd2fc"
    },
    {
      "cell_type": "code",
      "source": "clf = LogisticRegression()\nclf.fit(train_data, target)\ntest_data = test.drop(\"PassengerId\", axis=1)\npredict = clf.predict(test_data)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:17:05.754686Z",
          "iopub.status.busy": "2023-04-09T11:17:05.754258Z",
          "iopub.status.idle": "2023-04-09T11:17:05.778914Z",
          "shell.execute_reply": "2023-04-09T11:17:05.777884Z",
          "shell.execute_reply.started": "2023-04-09T11:17:05.754644Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "4186d80a"
    },
    {
      "cell_type": "markdown",
      "source": "로지스틱 회귀의 모델을 생성하여 점수를 측정해보았다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "abe8ac6c"
    },
    {
      "cell_type": "code",
      "source": "last = pd.DataFrame({\n    'PassengerId' : test['PassengerId'],\n    'Survived' : predict})\n\nlast.to_csv('last.csv', index=False)",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:23:25.219849Z",
          "iopub.status.busy": "2023-04-09T11:23:25.219386Z",
          "iopub.status.idle": "2023-04-09T11:23:25.228638Z",
          "shell.execute_reply": "2023-04-09T11:23:25.227063Z",
          "shell.execute_reply.started": "2023-04-09T11:23:25.219806Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "6f85d2bb"
    },
    {
      "cell_type": "markdown",
      "source": "예측 결과를 PassengerId와 매치시켜 데이터프레임으로 묶은 뒤 csv파일로 저장하였다.",
      "metadata": {
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "id": "c39e1714"
    },
    {
      "cell_type": "code",
      "source": "last = pd.read_csv(\"last.csv\")\nlast.head()",
      "metadata": {
        "execution": {
          "iopub.execute_input": "2023-04-09T11:23:27.266942Z",
          "iopub.status.busy": "2023-04-09T11:23:27.266492Z",
          "iopub.status.idle": "2023-04-09T11:23:27.282521Z",
          "shell.execute_reply": "2023-04-09T11:23:27.281047Z",
          "shell.execute_reply.started": "2023-04-09T11:23:27.266901Z"
        },
        "papermill": {
          "duration": null,
          "end_time": null,
          "exception": null,
          "start_time": null,
          "status": "pending"
        },
        "tags": []
      },
      "execution_count": null,
      "outputs": [],
      "id": "e26cef75"
    }
  ]
}