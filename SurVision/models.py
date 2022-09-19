#from market import db, login_manager
#from market import bcrypt
#from flask_login import UserMixin

### not implemented yet ###

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# This class needs to be adjusted for this use.
class User(db.Model, UserMixin):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(length=30), nullable=False, unique=True)
    email_address = db.Column(db.String(length=50), nullable=False, unique=True)
    password_hash = db.Column(db.String(length=60), nullable=False)
    budget = db.Column(db.Integer(), nullable=False, default=1000)
    # relationship, see below
    items = db.relationship('Item', backref='owned_user', lazy=True)

    @property
    def prettier_budget(self):
        if len(str(self.budget)) >= 4:
            return f'{str(self.budget)[:-3]},{str(self.budget)[-3:]}'
        else:
            return f'{self.budget}'

    @property
    def password(self):
        return self.password

    @password.setter
    def password(self, plain_text_password):
        self.password_hash = bcrypt.generate_password_hash(plain_text_password).decode('utf-8')

    def check_password_correction(self, attempted_password):
        return bcrypt.check_password_hash(self.password_hash, attempted_password)

    def can_purchase(self, item_obj):
        return self.budget >= item_obj.price

    def can_sell(self, item_obj):
        return item_obj in self.items

# This class is set up as key but still needs work.
class AnswerKeys(db.Model):
    key_name = db.Columns(db.String(length=30), nullable=False, unique=True)
    version = db.Columns(db.Integer(), nullable=False, unique=True)
    imageA = db.Columns(db.String(length=30), nullable=False, unique=True)
    imageB = db.Columns(db.String(length=30), nullable=False, unique=True)
    QA = db.Columns(db.String(length=30), nullable=False) # string will be converted to list
    QB = db.Columns(db.String(length=30), nullable=False) # string will be converted to list
    altanswer_choices = db.Columns(db.String(length=30)) # string will be converted to list
    altanswer_questions = db.Columns(db.String(length=30)) # string will be converted to list
    paper_size = db.Columns(db.Integer())
    orientation = db.Columns(db.String(length=30)) # string will be converted to list
    blank_thresh = db.Columns(db.Integer())

    def __repr__(self):
        return f'AnswerKey {self.key_name}'

