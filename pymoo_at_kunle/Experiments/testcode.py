
def removeDuplicate(a):   
    f = ""
    for e in set(a):
        f = f + e
    return f 

def freqTabulate(a):
    dico = dict()  #empty dictionary
    for e in removeDuplicate(a):  #traverse the unique chars in a
        dico[e] = a.count(e)    # add a count of unique chars in a to dictionary
    return dico

def  countNumberOfDeltetionsInBothStrings(a,b):
    #count of chars in a but not in b
    a = a.lower()
    b = b.lower()
    freq_a = freqTabulate(a)
    freq_b = freqTabulate(b)
    count = 0 # number of chars to delete in both a and b
    for key in freq_a.keys():
        if key not in b:
            count = count + freq_a[key]  
        else:   # the key/char occurs in both a and b
            count_a = freq_a[key]
            count_b =  freq_b[key]
            count = count + abs (count_a - count_b)
    for key in freq_b.keys():
        if key not in a:
            count+= freq_b[key]
    return count 


     

a = "A".lower()  
b =  "B".lower()

#print("final:", countNumberOfDeltetionsInBothStrings(a,b))





freq_a = freqTabulate(a)   # {"A":3,  "B": 2}

freq_b = freqTabulate(b)

#print(freq_a, freq_b)

 

'''
for key in freq_a.keys():
    print(freq_a,  key)
    if key in b:
        print(freq_a[key], freq_b[key])

'''

'''

#following is the api code for isooko beta site

f8e61fe79ecb93c190dbe73bc946a3531d181e57b1476b9f577733fc6400586b


'''


def sendEmail(subject = "None", receiver_email = "adekunleadekoya@gmail.com", message = "N/A"):


    import smtplib     

    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls 
    password =  "vosuycqrmnmhzcpt"     
    # Try to log in to server and send email
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465) 
        # Perform operations via server
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
        server.quit() 

    except Exception as e:
        # Print any error messages to stdout
        print("error: \n")
        print(e)
    finally:
        #server.quit() 
        pass

subject = "Isooko: account verification"   
linkUsed2VerifyAccount = "https://serieux-maison-90027.herokuapp.com/processor?is=x00"
message = "Kindly open the link below in a browser to verify your account...."
receiver_email = "adekunleadekoya@gmail.com"
sender_email = "adekunleadekoya@gmail.com"
message = f""" \n \
From:  {sender_email}     
To: adekunleadekoya@gmail.com
Subject: {subject}\n
{message}\n {linkUsed2VerifyAccount}
"""
#sendEmail(subject,receiver_email, message )

from cryptography.fernet import Fernet
key =  b'tIhLBaAgcxcrAKJmNNDdzOfXcQvUAbIEWg7eLweK3MQ='
f = Fernet(key)
bytedEmail =  receiver_email.encode() #converts string to bytes
token = f.encrypt(bytedEmail)
mi = token.decode()  #converts to string so as to fit into url string       
email = f.decrypt(mi.encode()).decode()  
print(email)