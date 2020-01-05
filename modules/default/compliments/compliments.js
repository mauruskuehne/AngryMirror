var count_same_res = 1;

var person = "Nobody";
Module.register("compliments", {

	person: "unknown",
	emotion: "sad",
	last_payload: "",
	// Module config defaults.
	defaults: {
		compliments: {
			neutral: [
				"You're a neutral looking person."
			],
			happy: [
				"Get out of my view!",
				"Why are you still here?",
				"Looking at you makes me sad."
			],
			sad: [
				"Hey there!",
				"good morning, handsome!",
				"Enjoy your day!",
				"how was your sleep?",
				"beautiful!",
				"you look sexy!",
				"you look good today!",
				"Wow you look hot!",
				"You look nice!",
				"sexy!"
			],
			anger: [
				"Do you know what’s cool? Winter!",
				"What fish is the best fighter? The Swordfish!",
			],
			surprise: [
				"Don't look so shocked, you look like this every day!"
			]
		},
		updateInterval: 30000,
		remoteFile: null,
		fadeSpeed: 4000,
		morningStartTime: 3,
		morningEndTime: 12,
		afternoonStartTime: 12,
		afternoonEndTime: 17
	},

	// Define required scripts.
	getScripts: function () {
		return ["moment.js"];
	},

	// Define start sequence.
	start: function () {
		Log.info("Starting module: " + this.name);
		this.person = "unknown";
		this.lastComplimentIndex = -1;

		var self = this;
		if (this.config.remoteFile !== null) {
			this.complimentFile(function (response) {
				self.config.compliments = JSON.parse(response);
				self.updateDom();
			});
		}

		self.sendSocketNotification("Inititate");

		// Schedule update timer.
		setInterval(function () {
			self.updateDom(self.config.fadeSpeed);
		}, this.config.updateInterval);
	},

	/* randomIndex(compliments)
	 * Generate a random index for a list of compliments.
	 *
	 * argument compliments Array<String> - Array with compliments.
	 *
	 * return Number - Random index.
	 */
	randomIndex: function (compliments) {
		if (compliments.length === 1) {
			return 0;
		}

		var generate = function () {
			return Math.floor(Math.random() * compliments.length);
		};

		var complimentIndex = generate();

		while (complimentIndex === this.lastComplimentIndex) {
			complimentIndex = generate();
		}

		this.lastComplimentIndex = complimentIndex;

		return complimentIndex;
	},

	/* complimentArray()
	 * Retrieve an array of compliments for the time of the day.
	 *
	 * return compliments Array<String> - Array with compliments for the time of the day.
	 */
	complimentArray: function () {
		var hour = moment().hour();
		var compliments;

		if (typeof compliments === "undefined") {
			compliments = new Array();
		}

		if(this.emotion == "neutral") {
			compliments.push.apply(compliments, this.config.compliments.neutral);
		}
		else if (this.emotion === "sad"){
			compliments.push.apply(compliments, this.config.compliments.sad);
		}
		else if (this.emotion === "happy"){
			compliments.push.apply(compliments, this.config.compliments.happy);
		}
		else if (this.emotion === "anger"){
			compliments.push.apply(compliments, this.config.compliments.anger);
			//compliments.push.apply(compliments, this.config.compliments.happy);
		} 
		else if (this.emotion === "surprise"){
			compliments.push.apply(compliments, this.config.compliments.surprise);
			//compliments.push.apply(compliments, this.config.compliments.happy);
		}

		return compliments;
	},

	/* complimentFile(callback)
	 * Retrieve a file from the local filesystem
	 */
	complimentFile: function (callback) {
		var xobj = new XMLHttpRequest(),
			isRemote = this.config.remoteFile.indexOf("http://") === 0 || this.config.remoteFile.indexOf("https://") === 0,
			path = isRemote ? this.config.remoteFile : this.file(this.config.remoteFile);
		xobj.overrideMimeType("application/json");
		xobj.open("GET", path, true);
		xobj.onreadystatechange = function () {
			if (xobj.readyState === 4 && xobj.status === 200) {
				callback(xobj.responseText);
			}
		};
		xobj.send(null);
	},

	/* complimentArray()
	 * Retrieve a random compliment.
	 *
	 * return compliment string - A compliment.
	 */
	randomCompliment: function () {
		var compliments = this.complimentArray();
		var index = this.randomIndex(compliments);

		return compliments[index];
	},

	// Override dom generator.
	getDom: function () {
		var complimentText = this.randomCompliment();
		if(this.emotion === "happy") {
			complimentText = "" + this.person + ", you're smiling... I think you are happy!\n" + complimentText;
		} else {
			complimentText = "" + this.person + ", I think you feel " + this.emotion + "!\n" + complimentText;
		}
		complimentText = complimentText.replace("Nobody", this.person);

		if (this.person === "nobody") {
			complimentText = "";
		}

		var compliment = document.createTextNode(complimentText);
		var wrapper = document.createElement("div");
		wrapper.className = this.config.classes ? this.config.classes : "thin xlarge bright pre-line";
		wrapper.appendChild(compliment);

		return wrapper;
	},

	// Override notification handler.
	notificationReceived: function (notification, payload, sender) {
		var that = this;
		Log.info("received notification", sender);
	},
	updatePerson: function (payload) {
		Log.info("received person", payload);
		let that = this;
		if (person === payload) {
			return;
		}

		that.person = payload;
		person = payload;

		MM.getModules().enumerate(function(module){
			if (typeof module.config.person == 'undefined') { //Module ohne Person immer anzeigen
				return
			}
			if(module.config.person === payload) { //statt Maurus -> payload
				Log.info("showing module", module.identifier);
				module.show(100);
			} else {
				Log.info("hiding module", module.identifier);
				module.hide(100);
			}
		})

		this.updateDom();
	},

	updateEmotion: function (payload) {
		Log.info("received emotion", payload);
		var that = this;
		if (that.emotion === payload) {
			return;
		}
		Log.info("reacting to new emotion...")
		that.emotion = payload;
		emotion = payload;
		//this.sendNotification("SPOTIFY_TRANSFER", "Jwos MacBook Pro")
		//this.sendNotification("SPOTIFY_PAUSE")
		if(emotion === "sad") {
			Log.info("playing sad playlist...")
			this.sendNotification("SHOW_PLAYLIST", "N3XdlnCknYM");
			this.sendNotification("SPOTIFY_PLAY", { 'context_uri': 'spotify:playlist:37i9dQZF1DX6J5NfMJS675' })
		}
		else if (emotion === "happy") {
			Log.info("playing happy playlist...")
			this.sendNotification("SHOW_PLAYLIST", "JoGeh850LbQ");
			this.sendNotification("SPOTIFY_PLAY", { 'context_uri': 'spotify:playlist:37i9dQZF1DXdPec7aLTmlC' })
		} else if (emotion === "anger") {
			Log.info("playing happy playlist...")
			this.sendNotification("SHOW_PLAYLIST", "LeHHhFuoN8Y");
			this.sendNotification("SPOTIFY_PLAY", { 'context_uri': 'spotify:playlist:37i9dQZF1DX4sWSpwq3LiO' })
		} else {
			Log.info("playing default playlist...")
			this.sendNotification("SHOW_PLAYLIST", "AgpWX18dby4");
			this.sendNotification("SPOTIFY_PLAY", { 'context_uri': 'spotify:playlist:37i9dQZF1DX6KItbiYYmAv' })
		}
		//this.sendNotification("SPOTIFY_PLAY")
		//this.sendNotification("SPOTIFY_NEXT")
		this.updateDom();
	},

	socketNotificationReceived: function (notification, payload) {
		var that = this;
		Log.info("received socket notification", payload);

		that.sendNotification("SHOW_ALERT", {
			type: "notification", 
			message: "" + payload.name + ", " + payload.emotion});
		if (notification !== 'person') {
			return;
		}
		if ( JSON.stringify(that.last_payload) === JSON.stringify(payload)) {
			count_same_res = count_same_res + 1;
			Log.info("received same payload", count_same_res);
		} else {
			count_same_res = 0;
			Log.info("received new payload", count_same_res);
		}
		that.last_payload = payload;

		if (count_same_res < 3) {
			return;
		}
		that.updatePerson(payload.name);
		that.updateEmotion(payload.emotion);

	},
});