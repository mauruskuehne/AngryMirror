/* Magic Mirror Config Sample
 *
 * By Michael Teeuw http://michaelteeuw.nl
 * MIT Licensed.
 *
 * For more information how you can configurate this file
 * See https://github.com/MichMich/MagicMirror#configuration
 *
 */

var config = {
	address: "localhost", // Address to listen on, can be:
	// - "localhost", "127.0.0.1", "::1" to listen on loopback interface
	// - another specific IPv4/6 to listen on a specific interface
	// - "", "0.0.0.0", "::" to listen on any interface
	// Default, when address config is left out, is "localhost"
	port: 8080,
	ipWhitelist: ["127.0.0.1", "::ffff:127.0.0.1", "::1"], // Set [] to allow all IP addresses
	// or add a specific IPv4 of 192.168.1.5 :
	// ["127.0.0.1", "::ffff:127.0.0.1", "::1", "::ffff:192.168.1.5"],
	// or IPv4 range of 192.168.3.0 --> 192.168.3.15 use CIDR format :
	// ["127.0.0.1", "::ffff:127.0.0.1", "::1", "::ffff:192.168.3.0/28"],

	language: "en",
	timeFormat: 24,
	units: "metric",

	modules: [
		{
			module: "alert",
			config: {
				position: "center",

				// The config property is optional.
				// See 'Configuration options' for more information.
			}
		},
		{
			module: "compliments",
			position: "bottom_center",
			
		},
		{
			module: 'custom/DailyXKCD',
			position: 'top_left',
			config: {
				person: "maurus",
				invertColors: true,
				showTitle: false,
				randomComic: true,
				showAltText: false
			}
		},
		{
			module: "custom/MMM-Spotify", // Path to youtube module from modules folder Exmaple: MagicMirror/modules/custom/MMM-EmbedYoutube/ so it's custom/MMM-EmbedYoutube
			position: "top_left",	// This can be any of the regions.
			config: {
				// See 'Configuration options' in README.md for more information.
				person: "jwo",
				loop: true,
				autoplay: true
			}
		},
		{
			module: "custom/MMM-Spotify", // Path to youtube module from modules folder Exmaple: MagicMirror/modules/custom/MMM-EmbedYoutube/ so it's custom/MMM-EmbedYoutube
			position: "top_left",	// This can be any of the regions.
			config: {
				// See 'Configuration options' in README.md for more information.
				person: "samuel",
				loop: true,
				autoplay: true
			}
		},
		{
			module: "custom/MMM-EmbedYoutube", // Path to youtube module from modules folder Exmaple: MagicMirror/modules/custom/MMM-EmbedYoutube/ so it's custom/MMM-EmbedYoutube
			position: "top_right",	// This can be any of the regions.
			config: {
				// See 'Configuration options' in README.md for more information.
				loop: true,
				autoplay: true
			}
		},
		{
			module: "custom/MMM-CalendarWeek",
			position: "bottom_center",	// This can be any of the regions. Best results in bottom region.
			config: {
				person: "maurus",
				calendars: [
					{
						url: 'https://outlook.office365.com/owa/calendar/6ca7ff1b73b04844bec8201abacd9869@innosolv.ch/34b4fe620e7d435abc3a6298e40226174452922640082790005/calendar.ics',
						symbol: 'calendar',
					},
				]
				// The config property is optional.
				// If no config is set, an example calendar is shown.
				// See 'Configuration options' for more information.
			}
		},
		{
			module: "custom/MMM-CalendarWeek",
			position: "bottom_center",	// This can be any of the regions. Best results in bottom region.
			config: {
				person: "jwo"
				// The config property is optional.
				// If no config is set, an example calendar is shown.
				// See 'Configuration options' for more information.
			}
		},
		{
			module: "custom/MMM-CalendarWeek",
			position: "bottom_center",	// This can be any of the regions. Best results in bottom region.
			config: {
				person: "samuel"
				// The config property is optional.
				// If no config is set, an example calendar is shown.
				// See 'Configuration options' for more information.
			}
		}
	]

};

/*************** DO NOT EDIT THE LINE BELOW ***************/
if (typeof module !== "undefined") { module.exports = config; }
