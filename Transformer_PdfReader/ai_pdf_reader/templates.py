
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user{
    background-color: #99a1b6
}
.chat-message .avatar {
  width: 15%;
}
.chat-message .avatar img {
  max_width: 78px;
  max-height: 78px;
  borer-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 85%;
  padding: 0 1.5rem;
}

'''
  
bot_template = '''
<div class="chat-message bot">
  <div class="avatar">
    <img src="https://i.ibb.co/qWBwpNb/Photo-logo-5.png">
  </div>
  <div class="message bot-message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
  <div class="avatar">
    <img src="https://i.ibb.co/5Gc9SPm/blacksheep.jpg">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''