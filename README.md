# Jack's Car Rental Problem from *Reinforcement Learning* by Sutton, Barto – 2nd Edition (2018)
This is a dynamic programming implementation of the "Jack's Car Rental" problem (Example 4.2) in Chapter 4.3 of the book.  
It's nice and fast, and treats the original, and modified problems (Exercise 4.7).
## Original Problem
### The policy on the orignal problem:   
![image](https://user-images.githubusercontent.com/28876473/171259071-e31214e0-01ca-486e-91cf-1f826700be73.png)  
### The state values on the original problem:  
![image](https://user-images.githubusercontent.com/28876473/171259098-80dd91f9-e2a2-49cc-89d0-724b90a63719.png)  
## Modified Problem from Exercise 4.7
Here, the first car moved from the first to the second location on any given day incurs no cost. But when storing more than ten cars in a location, that location incurs an extra cost of $4 for extra parking space that night.  
### The policy on the modified problem:  
![image](https://user-images.githubusercontent.com/28876473/171259136-dc749ba9-831b-4077-b35a-ca17ee25d216.png)  
### The state values on the modified problem:  
![image](https://user-images.githubusercontent.com/28876473/171259160-4cbc1edb-921e-41c7-b18c-2265aca8d986.png)  
