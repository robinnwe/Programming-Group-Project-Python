
# You start with 0 points which must be defined, so the points of the answers can be added

points = 0

# Introduction
print("Hey there, ")
print( "answer the following questions to see what type of investor you are. Do so by choosing between A, B or C. ")
print("Let's begin!")

# spacing, better layout
print()
print()

# Question 1, giving the user three different options to choose from, depending on the chosen answer, the points will be added
# \n for better layout
print("1. Just 60 days after you put money into an investment, its price falls 20%. Assuming none of the fundamentals have changed, what would you do? ")
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


print()

# Question 2a
print("2. Now look at the previous question another way. Your investment fell 20%, but it's part of a portfolio being used to meet investment goals with three different time horizons. ")
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
        
print()

# Question 2b
print("2b. What would you do if the goal were 15 years away? ")
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
             
print()

# Question 2c
print("2c. What would you do if the goal were 30 years away? ")
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
          
      
print()

# Question 3
print("3. The price of your retirement investment jumps 25% a month after you buy it. Again, the fundamentals haven't changed. After you finish gloating, what do you do? ")
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
    
  
print()

# Question 4
print("4. You're investing for retirement, which is 15 years away. Which would you rather do? ")
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
         
print()
      
# Question 5
print("5. You just won a big prize! But which one? It's up to you. ")
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
          

print()

# Question 6
print("6. A good investment opportunity just came along. But you have to borrow money to get in. Would you take out a loan? ")
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
         
      
print()
      
# Question 7
print("7. Your company is selling stock to its employees. In three years, management plans to take the company public. Until then, you won't be able to sell your shares and you will get no dividends. But your investment could multiply as much as 10 times when the company goes public. How much money would you invest? ")
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
        