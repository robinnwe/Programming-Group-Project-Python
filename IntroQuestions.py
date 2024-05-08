# You start with 0 points which must be defined, so the points of the answers can be added

points = 0

# Introduction
print("Hey there, ")
print( "answer the following questions to see what type of investor you are. Do so by choosing between A, B or C. ")
print("Let's begin!")

# Question 1, giving the user three different options to choose from, depending on the chosen answer, the points will be added
# \n for better layout
print("\n1. Just 60 days after you put money into an investment, its price falls 20%. Assuming none of the fundamentals have changed, what would you do? ")
print("\nA: Sell to avoid further worry and try something else, ") 
print("\nB: Do nothing and wait for the investment to come back, ")
print("\nC: Buy more. It was a good investment before; now it's a cheap investment, too ")
while True: 
    choice = input("\nenter your choice (A/B/C): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
        elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop 
        else: 
        print("Invalid choice. Please choose either A, B or C.")

# Question 2a
print("\n2. Now look at the previous question another way. Your investment fell 20%, but it's part of a portfolio being used to meet investment goals with three different time horizons. ")
print("\n2a. What would you do if the goal were five years away? ")
print( "\nA: Sell ")
print( "\nB: Do nothing")
print( "\nC: Buy more")
while True:
      choice = input("\nenter your choice (A/B/C): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
        elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        else:
        print("Invalid choice. Please choose either A, B or C.")


# Question 2b
print("\n2b. What would you do if the goal were 15 years away? ")
print( "\nA: Sell ")
print( "\nB: Do nothing ")
print( "\nC: Buy more")
while True:
      choice = input("\nenter your choice (A/B/C): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
        elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        else:
        print("Invalid choice. Please choose either A, B or C.")


# Question 2c
print("\n2c. What would you do if the goal were 30 years away? ")
print( "\nA: Sell ")
print( "\nB: Do nothing ")
print( "\nC: Buy more")
while True:
      choice = input("\nenter your choice (A/B/C): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
        elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        else:
        print("Invalid choice. Please choose either A, B or C.")
          

# Question 3
print("\n3. The price of your retirement investment jumps 25% a month after you buy it. Again, the fundamentals haven't changed. After you finish gloating, what do you do? ")
print( "\nA: Sell it and lock in your gains ")
print( "\nB: Stay put and hope for more gain ")
print( "\nC: Buy more; it could go higher")
while True:
      choice = input("\nenter your choice (A/B/C): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
        elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break#This will break the loop
        else:
        print("Invalid choice. Please choose either A, B or C.")

# Question 4
print("\n4. You're investing for retirement, which is 15 years away. Which would you rather do? ")
print( "\nA: Invest in a money-market fund or guaranteed investment contract, giving up the possibility of major gains, but virtually assuring the safety of your principal ")
print(" \nB: Invest in a 50-50 mix of bond funds and stock funds, in hopes of getting some growth, but also giving yourself some protection in the form of steady income ")
print( "\nC: Invest in aggressive growth mutual funds whose value will probably fluctuate significantly during the year, but have the potential for impressive gains over five or 10 years ")
while True:
      choice = input("\nenter your choice (A/B/C): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
        elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        else:
        print("Invalid choice. Please choose either A, B or C.")
      
    
# Question 5
print("\n5. You just won a big prize! But which one? It's up to you. ")
print( "\nA: $2,000 in cash ")
print( "\nB: A 50% chance to win $5,000") 
print( "\nC: A 20% chance to win $15,000")
while True: 
      choice = input("\nenter your choice (A/B/C): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
        elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        else:
        print("Invalid choice. Please choose either A, B or C.")
          

# Question 6
print("\n6. A good investment opportunity just came along. But you have to borrow money to get in. Would you take out a loan? ")
print( "\nA: Definitely not ")
print( "\nB: Perhaps ")
print( "\nC: yes")
while True:
      choice = input("\nenter your choice (A/B/C): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
        elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        else:
        print("Invalid choice. Please choose either A, B or C.")
         
        
# Question 7
print("\n7. Your company is selling stock to its employees. In three years, management plans to take the company public. Until then, you won't be able to sell your shares and you will get no dividends. But your investment could multiply as much as 10 times when the company goes public. How much money would you invest? ")
print( "\nA: None ") 
print( "\nB: Two months' salary ")
print( "\nC: Four months' salary")
while True:
    choice = input("\nenter your choice (A/B/C): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
        elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        else:
        print("Invalid choice. Please choose either A, B or C.")
        
        
# Question 8 (added questions to the form, only choosing between A and B)
print("\n8. Make a choice: ")
print( "\nA: a probability of 25% for a profit of CHF 30'000 ")
print("\nB: a probability of 20% for a profit of CHF 45'000")
while True:
      choice = input("\nenter your choice (A/B): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 1
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 2
        break #This will break the loop
        else:
        print("Invalid choice. Please choose either A or B.")
          

# Question 9 (added questions to the form, only choosing between A and B)
print("\n9. Make a choice: ")
print( "\nA: Profit of CHF 50'000 with a probability of 80% ")
print( "\nB: Profit of CHF 30'000 with a probability of 100%")
while True:
      choice = input("\nenter your choice (A/B): ")
        if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 1
        break
        elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 2
        break #This will break the loop
        else:
        print("Invalid choice. Please choose either A or B.")
          
# Introduction of the results
print("\nthank you for your answers")

# telling the user the amount of collected points
print("you have collected: ", points, "points")

# depending on the amout of points, the user will be assigned a type of investor
print("your investment type is: ")
if 0 <= points <= 3:
    print( "Type A = 8, extremely conservative")
if 4 <= points <= 7:
    print("Type A = 7, conservative investor")
if 8 <= points <= 11 :
    print("Type A = 6, conservative to moderate investor ")
if 12 <= points <= 15 :
    print("Type A = 5, moderate to aggressive investor")
if 16  <= points <= 19 :
    print("Type A = 4, aggressive investor")
if 20  <= points <= 22 :
    print(" Type A = 3, extremely aggressive investor")

