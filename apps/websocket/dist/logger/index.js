import winston from "winston";
import { env } from "../env";
const logger = winston.createLogger({
    level: env.LOG_LEVEL,
    format: winston.format.combine(winston.format.timestamp(), winston.format.errors({ stack: true }), winston.format.json()),
    defaultMeta: { service: "azuki-websocket" },
    transports: [
        new winston.transports.Console({
            format: env.NODE_ENV === "development"
                ? winston.format.combine(winston.format.colorize(), winston.format.simple())
                : winston.format.json(),
        }),
    ],
});
export default logger;
//# sourceMappingURL=index.js.map