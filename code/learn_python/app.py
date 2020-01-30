'''
Reading files
'''
# employee_file = open("employees.txt", "r")
#
# # print(employee_file.readable())
# # print(employee_file.readline())
# # print(employee_file.readlines())
# # print(employee_file.readlines()[1])
#
# for employee in employee_file.readlines():
#     print(employee)
#
# employee_file.close()

'''
Writing or appending files
'''

# # employee_file = open("employees.txt","a")
# # # 直接全部重写
# employee_file = open("employees.txt", "w")
# # employee_file.write("\nToby - Human Resources")
# # employee_file.write("\nKelly -Customer SERVIVE")
# employee_file.write("\nJim - ")
#
# employee_file.close()

'''
modules and pip
'''
# import Text
#
# print(Text.roll_dice(10))

'''
class and objects
'''
# from Student import Student
#
# student1 = Student("Jim", "Business", 3.1, False)
# student2 = Student("Pam", "Art", 2.5, True)
#
# print(student2.gpa)
'''
Building a Multiple Choize Quiz
'''
# from Question import Question
# question_prompts = [
#     "What color are apples?\n(a) Red/Green\n(b) Purple\n(c) Orange\n\n",
#     "What color are Bananas?\n(a) Teal\n(b) Magenta\n(c) Yellow\n\n",
#     "What color are Strawberries?\n(a) Yellow\n(b) Red\n(c) Blue\n\n"
# ]
#
# questions = [
#     Question(question_prompts[0], 'a'),
#     Question(question_prompts[1], 'c'),
#     Question(question_prompts[2], 'b')
# ]
#
# def run_test(questions):
#     score = 0
#     for question in questions:
#         answer = input(question.prompt)
#         if answer == question.answer:
#             score += 1
#     print("You got " + str(score) + "/" + str(len(questions)) + " correct!")
#
# run_test(questions)
'''
Object Funcitons
'''
# from Student import Student
# student1 = Student("Oscar", "Accounting", 3.21)
# student2 = Student("Phyllis", "Business", 3.8)
# print(student2.on_honor_roll())
'''
Inheritance
'''
# from Chef import Chef
# from ChineseChef import ChineseChef
#
# myChef = Chef()
# myChef.make_special_dish()
#
# myChineseChef = ChineseChef()
# myChineseChef.make_fried_rice()
# myChineseChef.make_special_dish()

