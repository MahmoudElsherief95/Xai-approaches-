[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/5APYwQLk)

# Explainable AI Assignment 2 - Model Explanations
In this assignment, you are challenged to explain a model. For this, you will research exisiting approaches and apply them to your model and interpret the results.

## General Information Submission

For the intermediate submission, please enter the group and dataset information. Coding is not yet necessary.

**Team Name:** crazy-pinnipeds

**Group Members**

| Student ID    | First Name  | Last Name      | E-Mail | Workload [%]  |
| --------------|-------------|----------------|--------|---------------|
| 51831270        | Markus      | Karner         |K51831270@students.jku.at  |25 %         |
| 11913261        | Hikmatullah      | Razaghi        |donhiki10@gmail.com  |25 %         |
| 12248150       | Mahmoud      | Elsherief         |mahmoudsherief2019@gmail.com  |25 %         |
| 12147764       | Kareem     | Muhammed    |  K12147764@students.jku.at|25 %         |


## Final Submission
The submission is done with this repository. Make to push your code until the deadline.

The repository has to include the implementations of the picked approaches and the filled out report in this README.

* Sending us an email with the code is not necessary.
* Update the *environment.yml* file if you need additional libraries, otherwise the code is not executeable.
* Save your final executed notebook(s) as html (File > Download as > HTML) and add them to your repository.

## Development Environment

Checkout this repo and change into the folder:
```
git clone https://github.com/jku-icg-classroom/xai_model_explanation_2022-crazy-pinnipeds.git
cd xai_model_explanation_2022-crazy-pinnipeds
```

Load the conda environment from the shared `environment.yml` file:
```
conda env create -f environment.yml
conda activate xai_model_explanation
```

> Hint: For more information on Anaconda and enviroments take a look at the README in our [tutorial repository](https://github.com/JKU-ICG/python-visualization-tutorial).

Then launch Jupyter Lab:
```
jupyter lab
```

Alternatively, you can also work with [binder](https://mybinder.org/), [deepnote](https://deepnote.com/), [colab](https://colab.research.google.com/), or any other service as long as the notebook runs in the standard Jupyter environment.


## Report

Please find the recording of a short presentation under this link: [Video Link](https://drive.google.com/file/d/1H1RX-DpjX7IVbHYGa7qGKmgvZFu6fUKf/view?usp=sharing)

### Model & Data

* Which model are you going to explain? What does it do? On which data is it used?
* From where did you get the model and the data used?
* Describe the model.

To show different explaination techniques we fitted a Random Forest on the Pima Indians Diabetes dataset from Kaggle.

  [Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)
  
  The dataset comprises 768 samples with 8 features and a binary class label.
  The columns in the Pima Indians Diabetes dataset represent the following:
  -  Pregnancies: Number of times pregnant
  -  Glucose: Plasma glucose concentration from a 2 hours oral glucose tolerance test
  -  BloodPressure: Diastolic blood pressure (mm Hg)
  -  SkinThickness: Triceps skinfold thickness (mm)
  -  Insulin: 2-Hour serum insulin (mu U/ml)
  -  BMI: Body mass index (weight in kg/(height in m)^2)
  -  DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
  - Age: Age in years
  - Outcome: Outcome variable (0 or 1), where 1 indicates the presence of diabetes and 0 indicates the absence.

We used 80% of the dataset to fit a Random Forest with 100 estimators. It achives an Accuracy of around 80%.

### Explainability Approaches
Find explainability approaches for the selected models. Repeat the section below for each approach to describe it.

#### Approach 1 - PDP

* Breifly summarize the approach.
  
  Partial dependence plots (PDP) show the dependence between the target response and a set of input features of interest, marginalizing over the values of all other input features (the 'complement' features). Intuitively, we can interpret the partial dependence as the expected target response as a function of the input features of interest.
    
  Due to the limits of human perception, the size of the set of input features of interest must be small (usually, one or two) thus the input features of interest are usually chosen among the most important features.
  
* Categorize this explainability approach according to the criteria by Hohman et al.
  
  WHY:
  * Model Interpretability & Feature Impact : PDP is employed to enhance the interpretability of machine learning models by illustrating the relationship between the target response and specific input features. It helps in understanding the impact of selected features on the model's predictions.
  * Explainability . PDP aids in explaining model predictions to model users, making it easier for non-experts to comprehend the model's behaviour and decision-making process.
  
  WHO:
  * Data scientists and model developers use PDP to analyse and interpret the behaviour of their models, gaining insights into the relationship between input features and the target response.
  * Non-experts : benefit from PDP as it provides a clear visual representation, making it accessible for individuals who may not have a technical background.
  
  WHAT:
  * PDP visualizes the relationship between the target response and selected input features while marginalizing over the values of other features , it shows how the target response changes with variations in the chosen features 
  
  HOW:
  * Aggregated Response Across Feature Values PDP calculates the aggregated response of the model across different values of the selected input features while keeping other features fixed. This is achieved by generating predictions for various combinations of feature values and then averaging the model's response.
  * Visualization through Line Plots :  PDP is often visualized through line plots, showing the trend of the target response as the values of the input features change..
  
  WHEN:
  * Model Analysis and Interpretation : PDP is commonly used during the model analysis phase, especially after the model is trained, to gain insights into the behavior of the model and understand the impact of specific features.
  
  WHERE:
  * Model Development and Validation PDP is applied in the context of model development and validation, where it helps in fine-tuning models and ensuring they align with the expected behaviour.
* Interpret the results here. How does it help to explain your model?

#### Approach 2 - ICE

* Breifly summarize the approach.

  Similar to a PDP, an individual conditional expectation (ICE) plot shows the dependence between the target function and an input feature of interest. However, unlike a PDP, which shows the average effect of the input feature, an ICE plot visualizes the dependence of the prediction on a feature for each sample separately with one line per sample. Due to the limits of human perception, only one input feature of interest is supported for ICE plots.

  While the PDPs are good at showing the average effect of the target features, they can obscure a heterogeneous relationship created by interactions. When interactions are present the ICE plot will provide many more insights.
  
* Categorize this explainability approach according to the criteria by Hohman et al.

  WHY:
  * Capturing Heterogeneous Relationships :  ICE plots are employed to capture and visualize heterogeneous relationships between the target function and an input feature. Unlike PDPs, ICE plots provide insights into individual sample variations, revealing nuances that might be obscured by averaging in PDPs.
  * Unveiling Interaction Effects : ICE plots are crucial when interactions between features and the target response exist. They uncover how each sample responds uniquely to changes in the chosen input feature, providing a more detailed understanding of complex relationships.
  
  WHO:
  * Non-experts. Researchers studying interaction effects within models benefit from ICE plots as they provide a granular view of how individual samples contribute to the overall model behaviour. 
  
  WHAT:
  * Individual Conditional Expectations :  ICE plots visualize the conditional expectations of the target response for each sample individually, illustrating how predictions change for a specific input feature.
  
  HOW:
  * Plotting Individual Sample Lines  plots represent the dependence between the target function and an input feature by plotting individual lines for each sample. Each line on the plot shows the predictions for a single sample as the input feature varies.
  * Addressing Interactions through individual lines : plots explicitly address interactions by providing a detailed view of how each sample responds to variations in the input feature, revealing patterns that might be overlooked in aggregate analyses. 
  
  WHEN:
  * Exploring Interactions in Model Output :  particularly useful when exploring models with intricate interactions, offering insights during the analysis phase after model training.
  
  WHERE:
  * In XAI class. This method (as well as the others described in these slides) were applied in the course of the Lab for XAI class.

#### Approach 3 - LIME

* Breifly summarize the approach.

  Local surrogate models are interpretable models that are used to explain individual predictions of black box machine learning models. Local interpretable model-agnostic explanations (LIME) is a paper in which the authors propose a concrete implementation of local surrogate models. Surrogate models are trained to approximate the predictions of the underlying black box model. Instead of training a global surrogate model, LIME focuses on training local surrogate models to explain individual predictions.
  
  The idea is quite intuitive. First, forget about the training data and imagine you only have the black box model where you can input data points and get the predictions of the model. You can probe the box as often as you want. Your goal is to understand why the machine learning model made a certain prediction. LIME tests what happens to the predictions when you give variations of your data into the machine learning model. LIME generates a new dataset consisting of perturbed samples and the corresponding predictions of the black box model. On this new dataset LIME then trains an interpretable model, which is weighted by the proximity of the sampled instances to the instance of interest. The interpretable model can be anything from the interpretable models, for example Lasso or a decision tree. The learned model should be a good approximation of the machine learning model predictions locally, but it does not have to be a good global approximation. 
  
* Categorize this explainability approach according to the criteria by Hohman et al.

  WHY:
  * Interpretability of Black Box Models : LIME is employed to enhance the interpretability of black box machine learning models by providing local, interpretable explanations for individual predictions. Understanding why a model made a specific prediction is crucial for building trust and making informed decisions. 
  * Local Surrogate Models for Transparency : LIME focuses on training local surrogate models, providing transparency at the individual prediction level. This approach enables users to grasp the decision-making process for a specific instance, offering insights that may not be apparent at a global level.
  
  WHO:
  * Data Scientist & Model Developers : Data scientists and model developers use LIME to interpret and explain individual predictions of black box models, facilitating model debugging and refinement.
  
  WHAT:
  * Local Surrogate Models  :  LIME implements local surrogate models that approximate the predictions of the black box model for individual instances. These models provide interpretable insights into why a specific prediction was made.
  
  HOW:
  * Probing Black Box Model Predictions: LIME simulates probing the black box model by generating a new dataset with perturbed samples and corresponding black box model predictions. This dataset is used to train local interpretable models, such as Lasso or decision trees, weighted by the proximity of the samples to the instance of interest.
  * Training Interpretable Models Locally  : LIME trains interpretable models locally, focusing on the neighborhood of the instance of interest. The interpretability is achieved by capturing the relationship between the perturbed samples and the corresponding black box model predictions in this local region..
  
  WHEN:
  * Individual Prediction Explanation  :  LIME is used when there is a need to explain individual predictions of a black box model, providing insights into the decision-making process for specific instances..
  
  WHERE:
  * In XAI class. This method (as well as the others described in these slides) were applied in the course of the Lab for XAI class.


#### Approach 4 - Shapely

* Breifly summarize the approach.

  Shapley values are used to fairly allocate the contribution of each feature to the prediction of a specific instance.
  The key idea behind Shapley values is to address the question: “How should the contribution or ‘credit’ for a specific prediction be distributed among the features?” Shapley values provide a way to fairly distribute the value of a coalition (a subset of features) among its members (individual features) based on their marginal contributions.
  
* Categorize this explainability approach according to the criteria by Hohman et al.

  WHY:
  * Interpretability & Explainability. Shapley values help users to see which features, i.e. pixels or parts of an image, helped the model to choose a label.
  * Teaching Deep Learning Concepts. This technique can also help non-experts to understand how deep learning works for image classification in general - models “scan” images to identify patterns that later will be used in classification.
  
  WHO:
  * Non-experts. This technique seems more suitable for non-experts by helping them understand what is happening inside “the black box”.
  
  WHAT:
  * Aggregated information. This technique shows the relationship between the input image and the learned features that the model considers most important.
  
  HOW:
  * Instance-based Analysis & Exploration. In our example, we calculate Shapley values for each individual instance in order to see which image regions are important for this specific image.
  * Analyzing Groups of Instances. Although we didn’t do it in this case, it is possible to apply this method to, for example, several images with the same label to see if similarities in how the model recognizes them.
  * Algorithms for Attribution & Feature Visualization. We applied an algorithm to see which parts of an image were more influential in categorizing the image.
  
  WHEN:
  * After training. The algorithm was applied on already trained model, images from the test set were used in the calculation.
  
  WHERE:
  * In XAI class. This method (as well as the others described in these slides) were applied in the course of the Lab for XAI class.


#### Approach 5 - DICE

* Breifly summarize the approach.

  Diverse Counterfactual Explanations (DICE) is a counterfactual explanation technique designed to explain predictions for individual instances. It is a local and model-agnostic method, making it compatible with any machine learning model. DICE provides a clear interpretation by answering the question: "What needs to be changed to produce the desired outcome?"
  
* Categorize this explainability approach according to the criteria by Hohman et al.

  Why:
  * Interpretability and Explainability of based on an instance of the data to gain local understanding of
  the model
  
  Who:
  * Model Developer & Builders or potential model users
  
  What:
  * Explaining how the change of some feature values of an instance has an influence in outcome class
  
  How:
  * This is achieved by generating diverse counterfactual explanations. Counterfactuals represent
  instances where certain feature values are altered to produce a different predicted class, typically the
  opposite of the original class. Analyzing these counterfactuals reveals the impact of individual
  features on the predicted outcome.
  
  When:
  * The interpretability and explanation process takes place after the model has been trained. Once the
  RandomForest model is trained, the prediction function is employed to generate counterfactuals.
  
  Where:
  * This information and methodology are documented and discussed in publications such as ArXiv and
  the Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency. Visit the
  link for more detail https://github.com/interpretml/DiCE

### Summary of Approaches
Write a brief summary reflecting on all approaches.

All five approaches where model agnostic so it would not have made a difference if we used another model. 
However, the simplicity of the Random Forest allowed us to try different approaches easily without having resource (GPU) issues.

The global approaches all pointed showed that the Random Forest learned to have Glucose and BMI as the most important features.
This is also nicely reflected in the local appraoches. Interestingly LIME showed a different order of the four most important features (globally) when an example was misclassified.

The local counterfactuals created by DICE where also very interesting as they sometimes created unrealistic changes, like getting a BMI of 0.1.

| Approach    | Global/Local  | Model Agnostic/Specific      | 
| --------------|-------------|----------------|
| PDP        | Global      | Agnostic         |
| ICE        | Local      | Agnostic        |
| LIME       | Local      | Agnostic         |
| SHAPLEY       | Global/Local     | Agnostic    |
| DICE       | Global/Local     | Agnostic    |
