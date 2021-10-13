from flask_ngrok import run_with_ngrok
from flask import Flask, request, make_response, jsonify
app = Flask(__name__)


#@app.route("/")
#def home():
 #return "<h1>Running Flask on Google Colab!</h1>"

# create a route for webhook
@app.route('/api/messages', methods=['POST'])
def webhook():
    print("webhook1",request)
    # build a request object
    req = request.get_json(force=True)
    print("webhook2",req)
    # fetch action from json
    act = req.get('queryResult').get('action')
    # return response
    print("hook")
    return make_response(jsonify(results(act)))
# function for responses
def results(act):
    
    print("hook11",act)
    # return a fulfillment response
    return {'fulfillmentText': 'Your potential accident level is : 2   Your accident level is : 1                  Thanks for the conversation. Take necessary precautions and work safely'}
# run the app
if __name__ == '__main__':
    app.run()

#run_with_ngrok(app) #starts ngrok when the app is run