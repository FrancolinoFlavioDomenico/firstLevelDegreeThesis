const fastify = require("fastify")({
    logger: {
        enabled: true,
        file: "log/node_log.txt",
    },
    bodyLimit: 100 * 1024 * 1024,
});
fastify.register(require("@fastify/multipart"), {
    limits: {
        fileSize: 500 * 1024 * 1024,
    },
    attachFieldsToBody: "keyValues",
});

const fs = require("node:fs");
const { pipeline } = require("node:stream/promises");

const { ethers } = require("ethers");
const provider = new ethers.JsonRpcProvider('http://127.0.0.1:8545/');

/*********************************************************
 blockchain connection and tesContract test route
 **********************************************************/

//----------connection test-----------
fastify.get("/isConnected", async function (request, reply) {
    const isConnected = await !!provider.getNetwork()
    reply.send({ isConnected: isConnected });
});


/*----------------------------------------------------------
                   real used api
----------------------------------------------------------*/

//Get blockchain address of selected client {:clientCid}
fastify.get("/address/client/:clientCid", async function (request, reply) {
    const response = await provider.listAccounts();
    return response[request.params.clientCid].address;
});

//write weights of client {:clientCid} at round {:round} into blockchain
fastify.post("/weights/write/:clientCid/:round", async (request, reply) => {

    try {
        const weights = request.body.weights
        const blockchianAddress = request.body.blockchianAddress
        console.log(weights)
        console.log(blockchianAddress)
        reply.send()
    } catch (e) {
        request.log.error({
            e,
            message: "write file on blockchain failed",
        });
        reply.log.error(e, `failed cid ${request.params.clientCid}`);
    }
});

//write weights of client {:clientCid} at round {:round}
fastify.post("/weights/check/:round/:clientCid", async (request, reply) => {
    /* if weigth < avgWeitg of previous round
            add clientCid to blacklist
            write weigt on blockchain
            write blacklist on blockchain
      */
    try {
        const data = await request.file({
            limits: { fileSize: 500 * 1024 * 1024 },
        });
        const buffer = await data.toBuffer();
        console.log(JSON.parse(buffer.toString()));
        return reply.send();
    } catch (e) {
        request.log.error({
            e,
            message: "write weight on blockchain failed",
        });
        reply.log.error(e, `failed cid ${request.params.clientCid}`);
    }
});

//Get blockchain blacklist
fastify.get("/blacklist", async function (request, reply) {
    return null;
});

/*----------------------------------------------------------
              start server
----------------------------------------------------------*/
// Run the server!
fastify.listen({ port: 3000 }, function (err, address) {
    if (err) {
        fastify.log.error(err);
        process.exit(1);
    }
    // Server is now listening on ${address}
});
