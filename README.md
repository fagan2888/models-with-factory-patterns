# models with factory pattern
EasyModel compiles multiple models together and is implemented using factory pattern. The structure is driven by JSON files rather than actual coding.

Currently, 
1) classification models with classification report, feature inspection, score, SHAP analysis, and decision threshold inspection are implemented. 
2) Regressors with VIF and residual checks(normal distribution, autocorrelation, constand variance)  
3) Basic preprocessors such as get_dummies, standard_scaler are also implemented.
