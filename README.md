# Jack's Car Rental Problem from *Reinforcement Learning* by Sutton, Barto – 2nd Edition (2018)
This is a value iteration implementation of the "Jack's Car Rental" problem (Example 4.2) in Chapter 4.3 of the book.  
This is the first time I've tried my hand at dynamic programming, and you can see my slow progress if you click through my commits.  
I did not plan my approach before going in, so it took me quite a while to get a fast (and correct) algorithm.  
In the process, I sped up computation by a factor of over 2,000, so that after precomputation of static values, it takes less than two seconds to find the optimal policy.  
## Original Problem
### The policy on the orignal problem:  
![image](https://user-images.githubusercontent.com/28876473/168423805-350d08ac-b45c-4ef9-a2c4-eba9b68a5c77.png)  
### The state values on the original problem:  
![image](https://user-images.githubusercontent.com/28876473/168423824-61780be8-c79d-4b6c-b973-0704ec42b86c.png)  
## Modified Problem from Exercise 4.7
Here, the first car moved from the first to the second location on any given day incurs no cost. But when storing more than ten cars in a location, that location incurs an extra cost of $4 for extra parking space that night.  
### The policy on the modified problem:  
![image](https://user-images.githubusercontent.com/28876473/168423924-4c8e23f8-78ea-415b-bb8e-cbb7eddce563.png)  
### The state values on the modified problem:  
![image](https://user-images.githubusercontent.com/28876473/168423927-8c2ce309-51f4-49ab-8908-6061e55430ae.png)  
