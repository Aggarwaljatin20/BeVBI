# BeVBI: Vehicle-invariant drive-by monitoring across multiple bridges through bootstrapping-enhanced unsupervised domain adaptation

This is the repository for the following paper. If you use this implementation, please cite our paper:

## Description:

![image](https://github.com/Aggarwaljatin20/BeVBI/assets/135164893/fd25070f-8605-435b-b1ea-5112cc415ad2)

Bridge health monitoring (BHM) is important to detect damages in the early stages to avoid loss of human life and any disruption in the continuous bridge operations. Drive-by vehicle-based BHM approaches provide more scalable monitoring as compared to manual inspection and fixed sensors on bridges. Each vehicle can pass multiple bridges and can be used for monitoring multiple bridges. In our prior work, we developed a method that can diagnose damage in multiple bridges while eliminating the burden of collecting labeled data from all the bridges. It is achieved by learning features that are sensitive to damage and invariant across bridges. However, in real-world scenarios, vehicles passing the bridge possess varying properties such as suspension system, driving speed, and vehicle mass. Since the vibration signal obtained from the vehicle depends on the vehicle's properties, these variabilities in vehicle properties lead to inaccurate damage prediction of the bridge even for the same damage state. 

To overcome these challenges, we introduce a Bootstrapping-enhanced Vehicle -Bridge-Invariant (BeVBI) approach for robust drive-by BHM. It reduces the vibration signal variation due to varying vehicle properties through bootstrapping-based mean estimations. Specifically, vibration signals obtained from  vehicles (with varying vehicle properties) passing the bridge are randomly aggregated with replacement (i.e., bootstrapping) and averaged. Based on the central limit theorem, averaging the aggregated signals (bootstrapped signals) reduces signal variability due to vehicle properties by the square root of the number of aggregated signals. Further, these bootstrapped signals are used to predict the damage on multiple bridges by adopting an unsupervised domain learning algorithm. The performance of the above approach is evaluated using a numerical vehicle bridge interaction dataset with two different bridges and 4800 drive-by vehicles having different dynamic properties and speeds. Our approach is successful in diagnosing multiple bridges while being robust to varying vehicle properties. It performs 1.45x better in the detection and localization of damage  and 1.75x better in the quantification of damage as compared to baseline methods (MCNN and HierMUD).

## Run the demo example with
>jupyter notebook BeVBI_demo.ipynb

## Contact Information:
>jatin08@stanford.edu
